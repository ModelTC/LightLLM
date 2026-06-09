import types


def test_mixin_pre_init_pops_and_shares_managers():
    from lightllm.models.base_mtp_model import BaseMTPModel

    main = types.SimpleNamespace(_cos_cached="cos", _sin_cached="sin", req_manager="rm", mem_manager="mm")

    obj = BaseMTPModel.__new__(BaseMTPModel)
    kvargs = {"main_model": main, "mtp_previous_draft_models": ["d0"], "other": 1}
    obj._pre_init(kvargs)
    assert obj.main_model is main
    assert obj.mtp_previous_draft_models == ["d0"]
    assert "main_model" not in kvargs and "mtp_previous_draft_models" not in kvargs

    obj._init_custom()
    obj._init_req_manager()
    obj._init_mem_manager()
    assert obj._cos_cached == "cos" and obj._sin_cached == "sin"
    assert obj.req_manager == "rm" and obj.mem_manager == "mm"

    assert BaseMTPModel.is_mtp_draft_model is True
