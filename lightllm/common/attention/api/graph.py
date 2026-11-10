class GraphBinding:
    """一个捕获实例的 provider 资源，不能挂在跨批次共享的 Backend 上。

    after_submit 在原生 Graph 提交后立即调用，不能先等待 Graph 完成；某些设备
    的 Graph 正在等待此方法提交的任务更新。返回的事件须覆盖所有辅助流工作。
    wait_for_update 只等待上次对主机 metadata 的使用，保证 prepare 可以安全改写。
    close 在相关设备工作完成后、原生 Graph 销毁前调用，并须可重复执行。
    """

    def wait_for_update(self):
        pass

    def before_replay(self):
        pass

    def after_submit(self):
        return ()

    def close(self):
        pass
