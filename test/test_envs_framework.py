"""
环境变量管理框架测试
"""

import os
import pytest
import warnings


class TestEnvVar:
    """测试 EnvVar 类"""

    def setup_method(self):
        """每个测试前重置环境"""
        from lightllm.utils.envs.core import EnvManager

        EnvManager.reset_instance()
        # 清理测试用的环境变量
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

    def test_string_var(self):
        """测试字符串类型变量"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_STRING",
            var_type=EnvVarType.STRING,
            default="default_value",
            description="Test string variable",
        )

        # 测试默认值
        assert var.get() == "default_value"

        # 测试设置值
        os.environ["TEST_STRING"] = "new_value"
        var.clear_cache()
        assert var.get() == "new_value"

    def test_int_var(self):
        """测试整数类型变量"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_INT",
            var_type=EnvVarType.INT,
            default=42,
            min_value=0,
            max_value=100,
        )

        assert var.get() == 42

        os.environ["TEST_INT"] = "50"
        var.clear_cache()
        assert var.get() == 50

        # 测试范围验证
        os.environ["TEST_INT"] = "150"
        var.clear_cache()
        with pytest.raises(ValueError, match="大于最大值"):
            var.get()

    def test_bool_var(self):
        """测试布尔类型变量"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_BOOL",
            var_type=EnvVarType.BOOL,
            default=False,
        )

        assert var.get() is False

        for true_value in ["1", "true", "TRUE", "on", "ON", "yes", "YES"]:
            os.environ["TEST_BOOL"] = true_value
            var.clear_cache()
            assert var.get() is True, f"Failed for {true_value}"

        for false_value in ["0", "false", "FALSE", "off", "no"]:
            os.environ["TEST_BOOL"] = false_value
            var.clear_cache()
            assert var.get() is False, f"Failed for {false_value}"

    def test_choices_validation(self):
        """测试选项验证"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_CHOICES",
            var_type=EnvVarType.STRING,
            default="A",
            choices=["A", "B", "C"],
        )

        assert var.get() == "A"

        os.environ["TEST_CHOICES"] = "B"
        var.clear_cache()
        assert var.get() == "B"

        os.environ["TEST_CHOICES"] = "D"
        var.clear_cache()
        with pytest.raises(ValueError, match="不在有效选项中"):
            var.get()

    def test_alias(self):
        """测试别名功能"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_NEW_NAME",
            var_type=EnvVarType.STRING,
            default="default",
            aliases=["TEST_OLD_NAME"],
        )

        # 使用旧名称设置
        os.environ["TEST_OLD_NAME"] = "from_alias"
        var.clear_cache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = var.get()
            assert value == "from_alias"
            assert len(w) == 1
            assert "已废弃" in str(w[0].message)

    def test_set_method(self):
        """测试 set 方法"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_SET",
            var_type=EnvVarType.INT,
            default=0,
        )

        var.set(100)
        assert os.environ["TEST_SET"] == "100"
        assert var.get() == 100

    def test_list_int_var(self):
        """测试整数列表类型"""
        from lightllm.utils.envs.core import EnvVar, EnvVarType

        var = EnvVar(
            name="TEST_LIST_INT",
            var_type=EnvVarType.LIST_INT,
            default=[],
        )

        assert var.get() == []

        os.environ["TEST_LIST_INT"] = "1, 2, 3"
        var.clear_cache()
        assert var.get() == [1, 2, 3]


class TestEnvGroup:
    """测试 EnvGroup 类"""

    def setup_method(self):
        from lightllm.utils.envs.core import EnvManager

        EnvManager.reset_instance()

    def test_add_and_get(self):
        """测试添加和获取变量"""
        from lightllm.utils.envs.core import EnvGroup, EnvVar, EnvVarType

        group = EnvGroup("test_group", "Test group description")

        var1 = EnvVar(name="TEST_VAR1", var_type=EnvVarType.STRING, default="v1")
        var2 = EnvVar(name="TEST_VAR2", var_type=EnvVarType.INT, default=42)

        group.add(var1)
        group.add(var2)

        assert group.get("TEST_VAR1") == var1
        assert group.get_value("TEST_VAR2") == 42
        assert len(group) == 2

    def test_to_dict(self):
        """测试导出为字典"""
        from lightllm.utils.envs.core import EnvGroup, EnvVar, EnvVarType

        group = EnvGroup("test", "Test")
        group.add(EnvVar(name="TEST_A", var_type=EnvVarType.STRING, default="a"))
        group.add(EnvVar(name="TEST_B", var_type=EnvVarType.INT, default=1))

        result = group.to_dict()
        assert result == {"TEST_A": "a", "TEST_B": 1}


class TestEnvManager:
    """测试 EnvManager 类"""

    def setup_method(self):
        from lightllm.utils.envs.core import EnvManager

        EnvManager.reset_instance()

    def test_singleton(self):
        """测试单例模式"""
        from lightllm.utils.envs.core import EnvManager

        m1 = EnvManager.get_instance()
        m2 = EnvManager.get_instance()
        assert m1 is m2

    def test_register_group(self):
        """测试注册组"""
        from lightllm.utils.envs.core import EnvManager, EnvGroup, EnvVar, EnvVarType

        manager = EnvManager.get_instance()
        group = EnvGroup("mygroup", "My test group")
        group.add(EnvVar(name="TEST_MGR_VAR", var_type=EnvVarType.STRING, default="test"))

        manager.register_group(group)

        assert "mygroup" in manager.list_groups()
        assert manager.get_value("TEST_MGR_VAR") == "test"

    def test_alias_resolution(self):
        """测试别名解析"""
        from lightllm.utils.envs.core import EnvManager, EnvGroup, EnvVar, EnvVarType

        manager = EnvManager.get_instance()
        group = EnvGroup("test", "Test")
        group.add(
            EnvVar(
                name="TEST_CANONICAL",
                var_type=EnvVarType.STRING,
                default="value",
                aliases=["TEST_ALIAS"],
            )
        )
        manager.register_group(group)

        # 通过别名获取
        var = manager.get_var("TEST_ALIAS")
        assert var.name == "TEST_CANONICAL"


class TestRegistry:
    """测试预定义注册表"""

    def test_logging_group(self):
        """测试日志组"""
        from lightllm.utils.envs.registry import env

        assert env.logging.LOG_LEVEL.get() == "info"
        assert env.logging.LOG_DIR.get() is None

    def test_server_group(self):
        """测试服务器组"""
        from lightllm.utils.envs.registry import env

        assert env.server.GUNICORN_TIMEOUT.get() == 180
        assert env.server.GUNICORN_KEEP_ALIVE.get() == 10

    def test_distributed_group(self):
        """测试分布式组"""
        from lightllm.utils.envs.registry import env

        assert env.distributed.MOE_MODE.get() == "TP"

    def test_backward_compat_alias(self):
        """测试向后兼容别名"""
        from lightllm.utils.envs.registry import env

        # 使用旧的拼写错误名称
        os.environ["LIGHTLMM_GUNICORN_TIME_OUT"] = "300"
        env.server.GUNICORN_TIMEOUT.clear_cache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = env.server.GUNICORN_TIMEOUT.get()
            assert value == 300
            # 应该有废弃警告
            assert any("已废弃" in str(warning.message) for warning in w)

        # 清理
        del os.environ["LIGHTLMM_GUNICORN_TIME_OUT"]

    def test_generate_docs(self):
        """测试文档生成"""
        from lightllm.utils.envs.registry import env

        docs = env.manager.generate_docs()
        assert "# LightLLM 环境变量参考" in docs
        assert "LIGHTLLM_LOG_LEVEL" in docs
        assert "LIGHTLLM_GUNICORN_TIMEOUT" in docs


class TestCompat:
    """测试兼容模块"""

    def setup_method(self):
        """每个测试前重置环境"""
        from lightllm.utils.envs.core import EnvManager
        from lightllm.utils.envs.registry import env

        # 清理可能残留的环境变量
        for key in ["LIGHTLMM_GUNICORN_TIME_OUT", "LIGHTLLM_GUNICORN_TIMEOUT"]:
            if key in os.environ:
                del os.environ[key]

        # 清除缓存
        env.server.GUNICORN_TIMEOUT.clear_cache()

    def test_deprecated_functions(self):
        """测试废弃函数发出警告"""
        from lightllm.utils.envs.compat import get_lightllm_gunicorn_time_out_seconds

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_lightllm_gunicorn_time_out_seconds()
            assert result == 180
            assert len(w) == 1
            assert "已废弃" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
