from .state import ComposedBoundedState


class ConvSSMState(ComposedBoundedState):
    """为 GDN/KDA 保留 Conv、SSM 和游标的兼容属性，数据由组合的 storage 持有。"""

    @property
    def conv_state(self):
        return self.resource.buffers.get("conv_state")

    @conv_state.setter
    def conv_state(self, value):
        self.resource.buffers["conv_state"] = value

    @property
    def ssm_state(self):
        return self.resource.buffers.get("ssm_state")

    @ssm_state.setter
    def ssm_state(self, value):
        self.resource.buffers["ssm_state"] = value

    @property
    def mtp_state_index(self):
        return self.resource.buffers.get("mtp_state_index")

    @mtp_state_index.setter
    def mtp_state_index(self, value):
        self.resource.buffers["mtp_state_index"] = value
