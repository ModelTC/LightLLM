import pytest

from lightllm.server.core.objs.out_token_circlequeue import CircularQueue, LIGHTLLM_OUT_TOKEN_QUEUE_SIZE


def test_queue_initialization():
    queue = CircularQueue()
    assert queue.head == 0
    assert queue.tail == 0
    assert len(queue) == 0
    assert queue.is_empty() is True
    assert queue.is_full() is False


def test_queue_push_and_pop():
    queue = CircularQueue()

    # Push an item
    queue.push("token1", 0, False, 1)
    assert len(queue) == 1
    assert queue.is_empty() is False
    assert queue.is_full() is False

    # Pop the item
    token, src_index, special, out_token_count = queue.pop()
    assert token == "token1"
    assert src_index == 0
    assert len(queue) == 0
    assert special is False
    assert out_token_count == 1


def test_queue_push_multiple_items():
    queue = CircularQueue()

    # Push multiple items
    queue.push("token1", 0, False, 1)
    queue.push("token2", 1, False, 2)
    queue.push("token3", 2, False, 3)

    assert len(queue) == 3
    assert queue.is_empty() is False
    if len(queue) == LIGHTLLM_OUT_TOKEN_QUEUE_SIZE - 1:
        assert queue.is_full() is True

    # Pop items
    token, src_index, special, out_token_count = queue.pop()
    assert token == "token1"
    assert src_index == 0
    assert special is False
    assert out_token_count == 1

    token, src_index, special, out_token_count = queue.pop()
    assert token == "token2"
    assert src_index == 1
    assert special is False
    assert out_token_count == 2

    token, src_index, special, out_token_count = queue.pop()
    assert token == "token3"
    assert src_index == 2
    assert special is False
    assert out_token_count == 3

    assert len(queue) == 0


def test_queue_full_condition():
    queue = CircularQueue()

    # Fill the queue
    for i in range(LIGHTLLM_OUT_TOKEN_QUEUE_SIZE - 1):
        queue.push(f"token{i}", i, False, 1)

    assert queue.is_full() is True

    # Try to push another item (should raise an exception)
    with pytest.raises(Exception, match="Queue is full"):
        queue.push("token_overflow", LIGHTLLM_OUT_TOKEN_QUEUE_SIZE, False, 1)


def test_queue_pop_empty_condition():
    queue = CircularQueue()

    # Attempt to pop from an empty queue (should raise an exception)
    with pytest.raises(Exception, match="Queue is empty"):
        queue.pop()


def test_queue_wrap_around():
    queue = CircularQueue()

    # Push enough items to wrap around
    for i in range(LIGHTLLM_OUT_TOKEN_QUEUE_SIZE - 1):
        queue.push(f"token{i}", i, False, i)

    # Pop all items
    for i in range(LIGHTLLM_OUT_TOKEN_QUEUE_SIZE - 1):
        token, src_index, _, _ = queue.pop()
        assert token == f"token{i}"
        assert src_index == i

    # Now the queue should be empty
    assert queue.is_empty() is True

    # Push again
    queue.push("token1", 0, False, 1)
    queue.push("token2", 1, False, 2)

    assert len(queue) == 2
    assert queue.is_empty() is False

    # Pop the items
    token, src_index, _, _ = queue.pop()
    assert token == "token1"
    assert src_index == 0

    token, src_index, _, _ = queue.pop()
    assert token == "token2"
    assert src_index == 1

    assert len(queue) == 0


# Run with pytest
if __name__ == "__main__":
    pytest.main()
