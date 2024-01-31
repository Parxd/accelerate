GRAD_ENABLED: bool = True


def is_grad_enabled():
    return GRAD_ENABLED


def set_grad_enabled(mode):
    global GRAD_ENABLED
    GRAD_ENABLED = mode


class NoGrad:
    def __init__(self):
        self.prev_state = is_grad_enabled()

    def __enter__(self):
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        set_grad_enabled(self.prev_state)
