class Tool:
    def run(self, *args, **kwargs): ...


class Pipeline(Tool):
    DEFAULT_FLOW = None

    def __init__(self, flow: list[Tool] | None = None):
        self.flow = self.DEFAULT_FLOW or flow

    def run(self, *args, **kwargs):
        state = args
        params = kwargs
        for tool in self.flow:
            state = tool.run(*state, **params)
