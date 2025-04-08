import base64
import httpx
from PIL import Image
from io import BytesIO


def image2base64(img_str: str):
    image_obj = Image.open(img_str)
    if image_obj.format is None:
        raise ValueError("No image format found.")
    buffer = BytesIO()
    image_obj.save(buffer, format=image_obj.format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def fetch_image(url, timeout, proxy=None):
    async with httpx.AsyncClient(proxy=proxy) as client:
        async with client.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            ans_bytes = []

            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                ans_bytes.append(chunk)
                # 接收的数据不能大于128M
                if len(ans_bytes) > 128:
                    raise Exception("image data is too big")

            content = b"".join(ans_bytes)
            return content
