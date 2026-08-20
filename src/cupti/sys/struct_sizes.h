enum CUpti_PCSampling_STRUCT_SIZES {
    PCSamplingConfigurationInfoParamsSize =  CUpti_PCSamplingConfigurationInfoParamsSize,
    PCSamplingEnableParamsSize =             CUpti_PCSamplingEnableParamsSize,
    PCSamplingDisableParamsSize =            CUpti_PCSamplingDisableParamsSize,
    PCSamplingStartParamsSize =              CUpti_PCSamplingStartParamsSize,
    PCSamplingStopParamsSize =               CUpti_PCSamplingStopParamsSize,
    PCSamplingGetNumStallReasonsParamsSize = CUpti_PCSamplingGetNumStallReasonsParamsSize,
    PCSamplingGetStallReasonsParamsSize =    CUpti_PCSamplingGetStallReasonsParamsSize,
    GetSassToSourceCorrelationParamsSize =   CUpti_GetSassToSourceCorrelationParamsSize,
    GetCubinCrcParamsSize =                  CUpti_GetCubinCrcParamsSize,
};

#ifdef _CUPTI_PMSAMPLING_H_
enum CUpti_PmSampling_STRUCT_SIZES {
    PmSampling_SetConfig_Params_STRUCT_SIZE =                   CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE,
    PmSampling_Enable_Params_STRUCT_SIZE =                      CUpti_PmSampling_Enable_Params_STRUCT_SIZE,
    PmSampling_Disable_Params_STRUCT_SIZE =                     CUpti_PmSampling_Disable_Params_STRUCT_SIZE,
    PmSampling_Start_Params_STRUCT_SIZE =                       CUpti_PmSampling_Start_Params_STRUCT_SIZE,
    PmSampling_Stop_Params_STRUCT_SIZE =                        CUpti_PmSampling_Stop_Params_STRUCT_SIZE,
    PmSampling_DecodeData_Params_STRUCT_SIZE =                  CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE,
    PmSampling_GetCounterAvailability_Params_STRUCT_SIZE =      CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE,
    PmSampling_GetCounterDataSize_Params_STRUCT_SIZE =          CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE,
    PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE = CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE,
    PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE =          CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE,
    PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE =   CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE,
};
#endif

#ifdef _CUPTI_PROFILER_HOST_H_
enum CUpti_Profiler_Host_STRUCT_SIZES {
    Profiler_Host_Initialize_Params_STRUCT_SIZE =                      CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE,
    Profiler_Host_Deinitialize_Params_STRUCT_SIZE =                    CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE,
    Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE =               CUpti_Profiler_Host_GetSupportedChips_Params_STRUCT_SIZE,
    Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE =                  CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE,
    Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE =                   CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE,
    Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE =             CUpti_Profiler_Host_GetMetricProperties_Params_STRUCT_SIZE,
    Profiler_Host_GetRangeName_Params_STRUCT_SIZE =                    CUpti_Profiler_Host_GetRangeName_Params_STRUCT_SIZE,
    Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE =             CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE,
    Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE =                CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE,
    Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE =              CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE,
    Profiler_Host_GetConfigImage_Params_STRUCT_SIZE =                  CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE,
    Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE =                  CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE,
    Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params_STRUCT_SIZE = CUpti_Profiler_Host_GetMaxNumHardwareMetricsPerPass_Params_STRUCT_SIZE,
#ifdef CUpti_Profiler_Host_GetSinglePassSets_Params_STRUCT_SIZE
    Profiler_Host_GetSinglePassSets_Params_STRUCT_SIZE =         CUpti_Profiler_Host_GetSinglePassSets_Params_STRUCT_SIZE,
#endif
#ifdef CUpti_Profiler_Host_GetMetricsInSinglePassSet_Params_STRUCT_SIZE
    Profiler_Host_GetMetricsInSinglePassSet_Params_STRUCT_SIZE = CUpti_Profiler_Host_GetMetricsInSinglePassSet_Params_STRUCT_SIZE,
#endif
#ifdef CUpti_Profiler_Host_SetDevicePartitionInfo_Params_STRUCT_SIZE
    Profiler_Host_SetDevicePartitionInfo_Params_STRUCT_SIZE =    CUpti_Profiler_Host_SetDevicePartitionInfo_Params_STRUCT_SIZE,
#endif
};
#endif

#ifdef _CUPTI_RANGE_PROFILER_H_
enum CUpti_RangeProfiler_STRUCT_SIZES {
    RangeProfiler_SetConfig_Params_STRUCT_SIZE =                   CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE,
    RangeProfiler_Enable_Params_STRUCT_SIZE =                      CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE,
    RangeProfiler_Disable_Params_STRUCT_SIZE =                     CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE,
    RangeProfiler_Start_Params_STRUCT_SIZE =                       CUpti_RangeProfiler_Start_Params_STRUCT_SIZE,
    RangeProfiler_Stop_Params_STRUCT_SIZE =                        CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE,
    RangeProfiler_PushRange_Params_STRUCT_SIZE =                   CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE,
    RangeProfiler_PopRange_Params_STRUCT_SIZE =                    CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE,
    RangeProfiler_DecodeData_Params_STRUCT_SIZE =                  CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE,
    RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE =          CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE,
    RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE = CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
    RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE =          CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE,
    RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE =    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE,
#ifdef CUpti_CUpti_RangeProfiler_GetDevicePartitionInfo_Params_STRUCT_SIZE
    RangeProfiler_GetDevicePartitionInfo_Params_STRUCT_SIZE =      CUpti_RangeProfiler_GetDevicePartitionInfo_Params_STRUCT_SIZE,
#endif
};
#endif

#ifdef _CUPTI_SASS_METRICS_H_
enum CUpti_SassMetrics_STRUCT_SIZES {
    SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE =  CUpti_SassMetrics_GetNumOfMetrics_Params_STRUCT_SIZE,
    SassMetrics_GetMetrics_Params_STRUCT_SIZE =       CUpti_SassMetrics_GetMetrics_Params_STRUCT_SIZE,
    SassMetrics_GetProperties_Params_STRUCT_SIZE =    CUpti_SassMetrics_GetProperties_Params_STRUCT_SIZE,
    SassMetricsSetConfig_Params_STRUCT_SIZE =         CUpti_SassMetricsSetConfig_Params_STRUCT_SIZE,
    SassMetricsUnsetConfig_Params_STRUCT_SIZE =       CUpti_SassMetricsUnsetConfig_Params_STRUCT_SIZE,
    SassMetricsEnable_Params_STRUCT_SIZE =            CUpti_SassMetricsEnable_Params_STRUCT_SIZE,
    SassMetricsDisable_Params_STRUCT_SIZE =           CUpti_SassMetricsDisable_Params_STRUCT_SIZE,
    SassMetricsGetDataProperties_Params_STRUCT_SIZE = CUpti_SassMetricsGetDataProperties_Params_STRUCT_SIZE,
    SassMetrics_InstanceValue_STRUCT_SIZE =           CUpti_SassMetrics_InstanceValue_STRUCT_SIZE,
    SassMetricsFlushData_Params_STRUCT_SIZE =         CUpti_SassMetricsFlushData_Params_STRUCT_SIZE,
};
#endif

#ifdef _CUPTI_PROFILER_TARGET_H_
enum CUpti_Profiler_STRUCT_SIZES {
    Profiler_Initialize_Params_STRUCT_SIZE =                                  CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
    Profiler_DeInitialize_Params_STRUCT_SIZE =                                CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
    Profiler_CounterDataImageOptions_STRUCT_SIZE =                            CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
    Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE =              CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
    Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE =                 CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
    Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
    Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE =    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
    Profiler_BeginSession_Params_STRUCT_SIZE =                                CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
    Profiler_EndSession_Params_STRUCT_SIZE =                                  CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
    Profiler_SetConfig_Params_STRUCT_SIZE =                                   CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
    Profiler_UnsetConfig_Params_STRUCT_SIZE =                                 CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
    Profiler_BeginPass_Params_STRUCT_SIZE =                                   CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
    Profiler_EndPass_Params_STRUCT_SIZE =                                     CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
    Profiler_EnableProfiling_Params_STRUCT_SIZE =                             CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
    Profiler_DisableProfiling_Params_STRUCT_SIZE =                            CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
    Profiler_IsPassCollected_Params_STRUCT_SIZE =                             CUpti_Profiler_IsPassCollected_Params_STRUCT_SIZE,
    Profiler_FlushCounterData_Params_STRUCT_SIZE =                            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
    Profiler_PushRange_Params_STRUCT_SIZE =                                   CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
    Profiler_PopRange_Params_STRUCT_SIZE =                                    CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
    Profiler_GetCounterAvailability_Params_STRUCT_SIZE =                      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
    Profiler_DeviceSupported_Params_STRUCT_SIZE =                             CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE,
};
#endif

#ifdef CUpti_ActivityConfig_STRUCT_SIZE
enum CUpti_Activity_STRUCT_SIZES {
    ActivityConfig_STRUCT_SIZE = CUpti_ActivityConfig_STRUCT_SIZE,
};
#endif

enum CUpti_Target_STRUCT_SIZES {
    Device_GetChipName_Params_STRUCT_SIZE = CUpti_Device_GetChipName_Params_STRUCT_SIZE,
};

#ifdef CUpti_SubscriberParams_STRUCT_SIZE
enum CUpti_Callbacks_STRUCT_SIZES {
    SubscriberParams_STRUCT_SIZE = CUpti_SubscriberParams_STRUCT_SIZE,
};
#endif