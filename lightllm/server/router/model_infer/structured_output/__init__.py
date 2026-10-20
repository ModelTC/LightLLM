"""Output constraints for the regular sampling pipeline.

HTTP workers compile grammars; InferenceContext caches loaded artifacts. InferReq owns grammar
progress and completion. Sampling fills fixed ReqSamplingParamsManager bitmask
slots after the existing post_handle handoff, and the postprocessing kernel reads
them by request index and MTP position.
"""
