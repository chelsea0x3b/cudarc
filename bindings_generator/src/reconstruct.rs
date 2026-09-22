//! Reconstructs per-version bindgen outputs from the committed merged
//! bindings, so adding a new CUDA version doesn't require regenerating
//! (and re-downloading) every already-supported version.
//!
//! [`merge`](crate::merge) is invertible: every item of a per-version
//! `sys_<version>.rs` appears verbatim in the merged file, either ungated
//! (identical across all versions) or behind a `#[cfg(any(feature = ...))]`
//! gate naming the versions it belongs to, and foreign functions keep their
//! exact signature inside the generated adapter fn. This module applies the
//! inverse mapping: select the items gated for one version, strip the gates,
//! and turn adapters back into `extern "C"` declarations.

use anyhow::{Context, Result, bail};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use syn::{Attribute, Item, Meta, punctuated::Punctuated};

use crate::ModuleConfig;
use crate::merge;
use crate::version::Version;

/// Seed `out/` with bindings reconstructed from `../src/<module>/sys/mod.rs`
/// for every supported version that is missing there and present in the
/// merged file. Returns the number of files written.
///
/// Versions absent from the merged file (i.e. newly added ones) are left for
/// bindgen to generate from the NVIDIA archives.
pub fn seed_from_merged(modules: &[ModuleConfig], cuda_versions: &[Version]) -> Result<usize> {
    let mut seeded = 0;
    for module in modules {
        let merged_path = Path::new("..")
            .join("src")
            .join(module.cudarc_name)
            .join("sys")
            .join("mod.rs");
        if !merged_path.exists() {
            continue;
        }

        let missing: Vec<Version> = module
            .versions(cuda_versions)
            .into_iter()
            .filter(|&v| !module.bindings_exist(v))
            .collect();
        if missing.is_empty() {
            continue;
        }

        let content = fs::read_to_string(&merged_path)?;
        let file = syn::parse_file(&content)
            .context(format!("Failed to parse {}", merged_path.display()))?;
        let gated_features = collect_gated_features(&file, module.feature_prefix);
        if gated_features.is_empty() {
            // Without any version gate we can't tell which versions the
            // merged file covers; leave them all to bindgen.
            log::info!(
                "{}: no version gates in merged bindings, skipping reconstruction",
                module.cudarc_name
            );
            continue;
        }

        for version in missing {
            let feature = version.feature_name(module.feature_prefix);
            if !gated_features.contains(&feature) {
                // Unknown to the merged file: a genuinely new version.
                continue;
            }
            let tokens = reconstruct_version(&file, &feature, module.feature_prefix);
            let out_path = module.bindings_path(version);
            fs::create_dir_all(out_path.parent().unwrap())?;
            // Written as raw tokens: these are merge inputs, and token
            // fidelity matters more than readability (reformatting can
            // alter macro tokens).
            fs::write(&out_path, tokens.to_string())?;
            seeded += 1;
        }
    }
    Ok(seeded)
}

/// Every version feature named by a `#[cfg]` gate anywhere in the file.
fn collect_gated_features(file: &syn::File, prefix: &str) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    for item in &file.items {
        if let Some(attrs) = item_attrs(item) {
            for attr in attrs {
                if let Some(gate) = version_gate(attr, prefix) {
                    features.extend(gate);
                }
            }
        }
    }
    features
}

/// Extract a single version's bindings from the merged file.
fn reconstruct_version(file: &syn::File, feature: &str, prefix: &str) -> TokenStream {
    let mut items = TokenStream::new();
    let mut foreign_fns = TokenStream::new();
    let oncelock_use = quote!(
        use std::sync::OnceLock;
    )
    .to_string();

    for item in &file.items {
        match item {
            // Merged-header boilerplate, not part of per-version bindings.
            Item::ExternCrate(_) => {}
            Item::Use(u) if u.to_token_stream().to_string() == oncelock_use => {}
            Item::Fn(f)
                if matches!(
                    f.sig.ident.to_string().as_str(),
                    "load" | "is_culib_present" | "culib"
                ) => {}

            // A function adapter built by `merge::build_adapter`: recover the
            // original `extern "C"` declaration from its signature.
            Item::Fn(f) => {
                if gate_admits(&f.attrs, feature, prefix) {
                    let mut sig = f.sig.clone();
                    sig.unsafety = None;
                    foreign_fns.extend(quote! { pub #sig; });
                }
            }

            other => {
                let Some(attrs) = item_attrs(other) else {
                    panic!("Unhandled item in merged bindings: {other:?}");
                };
                if gate_admits(attrs, feature, prefix) {
                    let mut other = other.clone();
                    strip_version_gate(item_attrs_mut(&mut other).unwrap(), prefix);
                    items.extend(other.into_token_stream());
                }
            }
        }
    }

    if foreign_fns.is_empty() {
        items
    } else {
        quote! {
            #items
            unsafe extern "C" {
                #foreign_fns
            }
        }
    }
}

/// True when the item has no version gate, or its gate names `feature`.
fn gate_admits(attrs: &[Attribute], feature: &str, prefix: &str) -> bool {
    attrs
        .iter()
        .find_map(|a| version_gate(a, prefix))
        .is_none_or(|gate| gate.iter().any(|f| f == feature))
}

fn strip_version_gate(attrs: &mut Vec<Attribute>, prefix: &str) {
    attrs.retain(|a| version_gate(a, prefix).is_none());
}

/// If `attr` is a `#[cfg(...)]` over version features of `prefix` (either a
/// bare `feature = "..."` or an `any(...)` list), return those features.
fn version_gate(attr: &Attribute, prefix: &str) -> Option<Vec<String>> {
    if !attr.path().is_ident("cfg") {
        return None;
    }
    let features = match attr.parse_args::<Meta>().ok()? {
        Meta::NameValue(nv) => vec![feature_value(&Meta::NameValue(nv))?],
        Meta::List(list) if list.path.is_ident("any") => list
            .parse_args_with(Punctuated::<Meta, syn::Token![,]>::parse_terminated)
            .ok()?
            .iter()
            .map(feature_value)
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    features
        .iter()
        .all(|f| is_version_feature(f, prefix))
        .then_some(features)
}

fn feature_value(meta: &Meta) -> Option<String> {
    if let Meta::NameValue(nv) = meta
        && nv.path.is_ident("feature")
        && let syn::Expr::Lit(lit) = &nv.value
        && let syn::Lit::Str(s) = &lit.lit
    {
        return Some(s.value());
    }
    None
}

fn is_version_feature(feature: &str, prefix: &str) -> bool {
    feature
        .strip_prefix(prefix)
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

fn item_attrs(item: &Item) -> Option<&Vec<Attribute>> {
    match item {
        Item::Const(i) => Some(&i.attrs),
        Item::Enum(i) => Some(&i.attrs),
        Item::Fn(i) => Some(&i.attrs),
        Item::Impl(i) => Some(&i.attrs),
        Item::Struct(i) => Some(&i.attrs),
        Item::Type(i) => Some(&i.attrs),
        Item::Union(i) => Some(&i.attrs),
        Item::Use(i) => Some(&i.attrs),
        _ => None,
    }
}

fn item_attrs_mut(item: &mut Item) -> Option<&mut Vec<Attribute>> {
    match item {
        Item::Const(i) => Some(&mut i.attrs),
        Item::Enum(i) => Some(&mut i.attrs),
        Item::Fn(i) => Some(&mut i.attrs),
        Item::Impl(i) => Some(&mut i.attrs),
        Item::Struct(i) => Some(&mut i.attrs),
        Item::Type(i) => Some(&mut i.attrs),
        Item::Union(i) => Some(&mut i.attrs),
        Item::Use(i) => Some(&mut i.attrs),
        _ => None,
    }
}

/// Compare bindings reconstructed from the merged files against the
/// per-version files currently in `out/`. Intended as a maintainer check
/// after a full generation run; fails on the first module that differs.
pub fn validate(modules: &[ModuleConfig], cuda_versions: &[Version]) -> Result<()> {
    let mut checked = 0;
    for module in modules {
        let merged_path = Path::new("..")
            .join("src")
            .join(module.cudarc_name)
            .join("sys")
            .join("mod.rs");
        if !merged_path.exists() {
            continue;
        }
        let content = fs::read_to_string(&merged_path)?;
        let file = syn::parse_file(&content)?;

        for version in module.versions(cuda_versions) {
            let out_path = module.bindings_path(version);
            if !out_path.exists() {
                continue;
            }
            let feature = version.feature_name(module.feature_prefix);
            let reconstructed = reconstruct_version(&file, &feature, module.feature_prefix);
            let reconstructed = merge::canonical_file(&reconstructed.to_string())?;
            let actual = merge::canonical_file(&fs::read_to_string(&out_path)?)?;

            let expected_items = item_map(&actual);
            let got_items = item_map(&reconstructed);
            for (key, expected) in &expected_items {
                match got_items.get(key) {
                    None => bail!("{} {version}: missing {key}", module.cudarc_name),
                    Some(got) if got != expected => bail!(
                        "{} {version}: mismatch in {key}:\n  expected: {expected}\n  got:      {got}",
                        module.cudarc_name
                    ),
                    Some(_) => {}
                }
            }
            for key in got_items.keys() {
                if !expected_items.contains_key(key) {
                    bail!("{} {version}: unexpected {key}", module.cudarc_name);
                }
            }
            checked += 1;
        }
    }
    println!("Validated {checked} reconstructed binding files");
    Ok(())
}

/// Flatten a file into comparable (kind + name) -> tokens entries.
/// `extern` blocks are flattened to their functions so block grouping
/// doesn't affect the comparison.
fn item_map(file: &syn::File) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for item in &file.items {
        match item {
            Item::ForeignMod(m) => {
                for item in &m.items {
                    if let syn::ForeignItem::Fn(f) = item {
                        map.insert(
                            format!("extern fn {}", f.sig.ident),
                            f.to_token_stream().to_string(),
                        );
                    }
                }
            }
            Item::Impl(i) => {
                map.insert(format!("impl {i:?}"), i.to_token_stream().to_string());
            }
            other => {
                let kind_and_name = match other {
                    Item::Const(i) => format!("const {}", i.ident),
                    Item::Enum(i) => format!("enum {}", i.ident),
                    Item::Struct(i) => format!("struct {}", i.ident),
                    Item::Type(i) => format!("type {}", i.ident),
                    Item::Union(i) => format!("union {}", i.ident),
                    Item::Use(i) => format!("use {}", i.to_token_stream()),
                    _ => format!("{other:?}"),
                };
                map.insert(kind_and_name, other.to_token_stream().to_string());
            }
        }
    }
    map
}
