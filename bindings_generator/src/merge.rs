use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use syn::{
    FnArg, ForeignItemFn, Item, ItemConst, ItemEnum, ItemFn, ItemImpl, ItemStruct, ItemType,
    ItemUnion, ItemUse, Pat,
};

use crate::ModuleConfig;
use crate::version::Version;

/// Build a unified adapter for a single foreign function.
///
/// The emitted adapter has both link strategies inside its body, gated on the
/// `dynamic-loading` feature: a per-symbol `OnceLock` that resolves the symbol
/// on first call when dynamic loading is enabled, or a plain `extern "C"` decl
/// + call when it's not. The outer `pub unsafe fn` and its signature are the
/// same in both modes, so callers don't see a difference.
fn build_adapter(
    func: &ForeignItemFn,
    versions: &[&Version],
    n_versions: usize,
    feature_prefix: &str,
) -> ItemFn {
    let features = versions
        .iter()
        .map(|v| v.feature_name(feature_prefix))
        .collect::<Vec<_>>();
    let feature_tok = if versions.len() == n_versions {
        quote! {}
    } else {
        quote! {
            #[cfg(any(#(feature=#features),*))]
        }
    };
    let sig = &func.sig;
    let fn_name = &sig.ident;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let arg_names: Vec<_> = inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg
                && let Pat::Ident(pat_ident) = &*pat_type.pat
            {
                return Some(pat_ident.ident.clone());
            }
            None
        })
        .collect();
    let symbol_cstr = cstr_expr(fn_name.to_string());
    let symbol_str = fn_name.to_string();

    let tokens = quote! {
        #feature_tok
        pub unsafe fn #fn_name(#inputs) #output {
            #[cfg(feature = "dynamic-loading")]
            let __result = {
                type __SymFn = unsafe extern "C" fn(#inputs) #output;
                static SYM: std::sync::OnceLock<__SymFn> = std::sync::OnceLock::new();
                let f = SYM.get_or_init(|| unsafe {
                    *culib()
                        .get::<__SymFn>(#symbol_cstr)
                        .unwrap_or_else(|e| panic!("Failed to load symbol {}: {e}", #symbol_str))
                });
                f(#(#arg_names),*)
            };
            #[cfg(not(feature = "dynamic-loading"))]
            let __result = {
                extern "C" {
                    fn #fn_name(#inputs) #output;
                }
                #fn_name(#(#arg_names),*)
            };
            __result
        }
    };
    syn::parse2(tokens).unwrap()
}

pub fn cstr_expr(mut string: String) -> TokenStream {
    string.push('\0');
    let b = proc_macro2::Literal::byte_string(string.as_bytes());
    quote! {
        #b
    }
}

#[derive(Debug)]
struct FunctionInfo<T> {
    declarations: BTreeMap<Version, T>,
}

impl<T> Default for FunctionInfo<T> {
    fn default() -> Self {
        Self {
            declarations: BTreeMap::new(),
        }
    }
}

impl<T> FunctionInfo<T> {
    fn insert(&mut self, version: &Version, value: T) -> Option<T> {
        self.declarations.insert(*version, value)
    }
}

#[derive(Default)]
struct BindingMerger {
    functions: BTreeMap<String, FunctionInfo<ForeignItemFn>>,
    enums: BTreeMap<String, FunctionInfo<ItemEnum>>,
    impls: BTreeMap<String, FunctionInfo<ItemImpl>>,
    structs: BTreeMap<String, FunctionInfo<ItemStruct>>,
    types: BTreeMap<String, FunctionInfo<ItemType>>,
    uses: BTreeMap<String, FunctionInfo<ItemUse>>,
    unions: BTreeMap<String, FunctionInfo<ItemUnion>>,
    consts: BTreeMap<String, FunctionInfo<ItemConst>>,

    lib_names: Vec<String>,
    n_versions: usize,
    feature_prefix: String,
}

impl BindingMerger {
    pub fn new(lib_names: Vec<String>, feature_prefix: String) -> Self {
        Self {
            lib_names,
            n_versions: 0,
            feature_prefix,
            ..Default::default()
        }
    }

    pub fn process_file(&mut self, path: &Path, version: &Version) -> Result<()> {
        self.n_versions += 1;
        let content = std::fs::read_to_string(path)?;
        let file = syn::parse_file(&content)?;

        for item in file.items {
            match item {
                Item::ForeignMod(foreign_mod) => {
                    for item in foreign_mod.items {
                        match &item {
                            syn::ForeignItem::Fn(func) => {
                                let name = func.sig.ident.to_string();
                                self.functions
                                    .entry(name)
                                    .or_default()
                                    .insert(version, func.clone());
                            }
                            other => println!(
                                "WARNING: Unhandled foreign item {other:?} in {path:?}... SKIPPING"
                            ),
                        }
                    }
                }
                Item::Struct(st) => {
                    let name = st.ident.to_string();
                    self.structs.entry(name).or_default().insert(version, st);
                }
                Item::Type(typ) => {
                    let name = typ.ident.to_string();
                    self.types.entry(name).or_default().insert(version, typ);
                }
                Item::Impl(imp) => {
                    let name = format!("{imp:?}");
                    self.impls.entry(name).or_default().insert(version, imp);
                }
                Item::Enum(en) => {
                    let name = en.ident.to_string();
                    self.enums.entry(name).or_default().insert(version, en);
                }
                Item::Use(us) => {
                    let name = format!("{us:?}");
                    self.uses.entry(name).or_default().insert(version, us);
                }
                Item::Union(un) => {
                    let name = un.ident.to_string();
                    self.unions.entry(name).or_default().insert(version, un);
                }
                Item::Const(con) => {
                    let name = con.ident.to_string();
                    self.consts.entry(name).or_default().insert(version, con);
                }
                other_item => {
                    panic!("Unhandled item {other_item:?}");
                }
            }
        }

        Ok(())
    }

    pub fn generate_unified_bindings(&self) -> TokenStream {
        let enums = self.write_to_output(&self.enums).expect("Write to output");
        let impls = self.write_to_output(&self.impls).expect("Write to output");
        let structs = self
            .write_to_output(&self.structs)
            .expect("Write to output");
        let types = self.write_to_output(&self.types).expect("Write to output");
        let uses = self.write_to_output(&self.uses).expect("Write to output");
        let unions = self.write_to_output(&self.unions).expect("Write to output");
        let consts = self.write_to_output(&self.consts).expect("Write to output");

        let lib_names = &self.lib_names;

        let adapters = self
            .create_unified_adapters(&self.functions)
            .expect("Write to output");

        TokenStream::from(quote! {
            // AUTOGENERATED UNIFIED CUDA BINDINGS
            // This file combines bindings from multiple CUDA versions
            #![cfg_attr(feature = "no-std", no_std)]
            #![allow(non_camel_case_types)]
            #![allow(non_snake_case)]
            #![allow(dead_code)]

            #[cfg(feature = "no-std")]
            extern crate alloc;
            #[cfg(feature = "no-std")]
            extern crate no_std_compat as std;

            #uses

            #consts

            #types

            #enums

            #structs

            #impls

            #unions

            #adapters

            #[cfg(feature = "dynamic-loading")]
            pub unsafe fn is_culib_present() -> bool {
                let lib_names = [#(#lib_names),*];
                let choices = lib_names
                    .iter()
                    .map(|l| crate::get_lib_name_candidates(l))
                    .flatten();
                for choice in choices {
                    if ::libloading::Library::new(choice).is_ok() {
                        return true;
                    }
                }
                false
            }

            #[cfg(feature = "dynamic-loading")]
            pub unsafe fn culib() -> &'static ::libloading::Library {
                static LIB: std::sync::OnceLock<::libloading::Library> = std::sync::OnceLock::new();
                LIB.get_or_init(|| {
                    let lib_names = std::vec![#(#lib_names),*];
                    let choices: std::vec::Vec<_> = lib_names
                        .iter()
                        .map(|l| crate::get_lib_name_candidates(l))
                        .flatten()
                        .collect();
                    for choice in choices.iter() {
                        if let Ok(lib) = ::libloading::Library::new(choice) {
                            return lib;
                        }
                    }
                    crate::panic_no_lib_found(lib_names[0], &choices);
                })
            }
        })
    }

    fn write_to_output<T: ToTokens + PartialEq<T>>(
        &self,
        info: &BTreeMap<String, FunctionInfo<T>>,
    ) -> Result<TokenStream> {
        let mut output = TokenStream::new();
        for (name, info) in info {
            let mut prev_decl: Option<&T> = None;
            let mut versions: Vec<Version> = vec![];
            for (version, decl) in &info.declarations {
                if let Some(prev_decl) = prev_decl
                    && prev_decl != decl
                {
                    if !versions.is_empty() {
                        log::debug!("Breaking change detected in {version} for {name}");
                    }
                    let features = versions
                        .iter()
                        .map(|v| v.feature_name(&self.feature_prefix))
                        .collect::<Vec<_>>();
                    output.extend(quote! {
                        #[cfg(any(#(feature = #features), *))]
                        #prev_decl
                    });
                    versions.clear();
                }
                versions.push(*version);
                prev_decl = Some(decl);
            }
            if !versions.is_empty() {
                if let Some(decl) = prev_decl {
                    if versions.len() == self.n_versions {
                        output.extend(decl.into_token_stream());
                    } else {
                        let features = versions
                            .iter()
                            .map(|v| v.feature_name(&self.feature_prefix))
                            .collect::<Vec<_>>();
                        output.extend(quote! {
                            #[cfg(any(#(feature = #features),*))]
                            #decl
                        });
                    }
                } else {
                    panic!("Previous version shouldn't be empty");
                }
            } else {
                panic!("Versions shouldn't be empty");
            }
        }
        Ok(output)
    }

    fn create_unified_adapters(
        &self,
        info: &BTreeMap<String, FunctionInfo<ForeignItemFn>>,
    ) -> Result<TokenStream> {
        let mut adapters: Vec<ItemFn> = vec![];
        for info in info.values() {
            let mut prev_decl: Option<&ForeignItemFn> = None;
            let mut versions = vec![];
            for (version, decl) in &info.declarations {
                if let Some(prev_decl) = prev_decl
                    && prev_decl != decl
                {
                    adapters.push(build_adapter(
                        prev_decl,
                        &versions,
                        self.n_versions,
                        &self.feature_prefix,
                    ));
                    versions.clear();
                }
                versions.push(version);
                prev_decl = Some(decl);
            }
            if !versions.is_empty()
                && let Some(decl) = prev_decl
            {
                adapters.push(build_adapter(
                    decl,
                    &versions,
                    self.n_versions,
                    &self.feature_prefix,
                ));
            }
        }

        Ok(quote! {
            #(#adapters)*
        })
    }
}

pub fn merge<P: AsRef<Path>>(
    binding_dir: P,
    output_filename: P,
    lib_names: Vec<String>,
    feature_prefix: &str,
    multi_progress: &MultiProgress,
) -> Result<()> {
    let binding_dir = binding_dir.as_ref();
    let module_name = binding_dir
        .components()
        .nth(1)
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("unknown");

    let entries: Vec<_> = fs::read_dir(binding_dir)?.collect::<std::io::Result<_>>()?;

    let pb = multi_progress.add(ProgressBar::new(entries.len() as u64));
    pb.set_style(ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len}")?);
    pb.set_message(format!("merge {module_name}"));

    let mut merger = BindingMerger::new(lib_names, feature_prefix.to_string());
    for entry in entries {
        let path = entry.path();
        if path.is_file() {
            let version = parse_version_from_filename(&path)?;
            merger.process_file(&path, &version)?;
        }
        pb.inc(1);
    }

    let unified = merger.generate_unified_bindings();
    let parsed = syn::parse2(unified.clone())
        .with_context(|| format!("In module {:?}", binding_dir.display()))?;
    std::fs::write(&output_filename, prettyplease::unparse(&parsed))?;
    pb.finish_with_message(format!("done  {module_name}"));
    Ok(())
}

fn parse_version_from_filename(path: &Path) -> Result<Version> {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .with_context(|| format!("non-utf8 filename: {}", path.display()))?;
    let version_str = stem
        .strip_prefix("sys_")
        .with_context(|| format!("expected 'sys_' prefix in {}", path.display()))?;
    version_str
        .parse()
        .with_context(|| format!("parsing version from {}", path.display()))
}

pub fn merge_bindings(modules: &[ModuleConfig]) -> Result<()> {
    let multi_progress = MultiProgress::new();
    modules.par_iter().try_for_each(|config| {
        merge(
            format!("out/{}/sys/linked", config.cudarc_name),
            format!("../src/{}/sys/mod.rs", config.cudarc_name),
            config.libs.iter().map(|&s| s.into()).collect(),
            config.feature_prefix,
            &multi_progress,
        )
    })
}
