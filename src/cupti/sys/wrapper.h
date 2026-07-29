#include <stdbool.h>
#include "cupti.h"
/*
#if __has_include("cupti_checkpoint.h")
# include "cupti_checkpoint.h"
#endif
#include "cupti_pcsampling_util.h"
*/
#include "cupti_pcsampling.h"
#if __has_include("cupti_pmsampling.h")
# include "cupti_pmsampling.h"
#endif
#if __has_include("cupti_profiler_host.h")
# include "cupti_profiler_host.h"
#endif
#if __has_include("cupti_range_profiler.h")
# include "cupti_range_profiler.h"
#endif
#if __has_include("cupti_sass_metrics.h")
# include "cupti_sass_metrics.h"
#endif
#include "cupti_target.h"

#include "struct_sizes.h"
