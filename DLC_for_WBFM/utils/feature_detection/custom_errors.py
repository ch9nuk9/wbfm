

class AnalysisOutOfOrderError(Exception):

    def __init__(self, required_step, attempted_step):
        self.required_step = required_step
        self.attempted_step = attempted_step
        self.message = f"Attempted step {attempted_step}, but this requires step {required_step} to be executed first"

        super().__init__(self.message)


class OverwritePreviousAnalysisError(Exception):

    def __init__(self, fieldname):
        self.fieldname = fieldname
        self.message = f"Should not overwrite field {fieldname}; if this was intended, then set this field to 'None'"

        super().__init__(self.message)


class DataSynchronizationError(Exception):

    def __init__(self, field1, field2, suggested_method=""):
        self.field1 = field1
        self.field2 = field2
        self.suggested_method = suggested_method
        self.message = f"Fields {field1} and {field2} should be synchronized"
        if len(suggested_method) > 0:
            self.message += f"; try {suggested_method}"

        super().__init__(self.message)
