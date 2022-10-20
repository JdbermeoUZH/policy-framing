from preprocessing.BOWPipeline import BOWPipeline


class Logger:
    def __init__(
            self,
            unit_of_analysis: str,
            preprocessing_pipeline: BOWPipeline
    ):

        self.unit_of_analysis = unit_of_analysis
        self.preprocessing_pipeline = preprocessing_pipeline

    def log_model_wide_performance(self, outer_fold_results):
        return 0

    def log_hyper_param_performance_outer_fold(self, outer_fold_results):
        return 0

    def log_hyper_param_performance_inner_fold(self, outer_fold_results):
        return 0