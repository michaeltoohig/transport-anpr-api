# from app.tests.utils.utils import random_lower_string
from fastapi.testclient import TestClient
# from sqlalchemy.orm import Session

# from app.config import settings
# from app.tests.utils.image import create_random_business_image, create_random_image, create_random_activity_image
# from app.tests.utils.business import create_random_business


def test_upload_image(client: TestClient) -> None:
    with open("app/tests/vuplate.jpg", "rb") as f:
        file = {"image": ("vuplate.jpg", f, "image/jpeg")}
        response = client.post(
            f"/detect/vehicles", files=file, data={"token": "test"}
        )
        assert response
        content = response.json()


def test_upload_image_read_status(client: TestClient) -> None:
    with open("app/tests/vuplate.jpg", "rb") as f:
        file = {"image": ("vuplate.jpg", f, "image/jpeg")}
        response = client.post(
            f"/detect/vehicles", files=file, data={"token": "test"}
        )
    assert response.status_code  # == 202
    content = response.json() 
    key = content["key"]
    print(key)
    response = client.get(
        f"/detect/vehicles/{key}"
    )
    assert response.status_code == 200
    content = response.json()
    assert content["status"] == "gotem"

# def test_plate_prediction(
#     client: TestClient
# ) -> None:
#     # superuser_token_headers["Content-Length"] = "500"
#     with open("app/tests/vuplate.jpg", "rb") as f:
#         file = {"image": ("vuplate.jpg", f, "image/jpeg")}
#         response = client.post(
#             f"/plate", files=file, data={"token": "test"}
#         )
#     assert response.status_code == 200
#     content = response.json()
#     assert content["prediction"]  # == "3953"  # should be "B3953" but have to figure how to detect the "B" in negative


# def test_plate_prediction_fails(
#     client: TestClient
# ) -> None:
#     # superuser_token_headers["Content-Length"] = "500"
#     with open("app/tests/test.jpg", "rb") as f:
#         file = {"image": ("test.jpg", f, "image/jpeg")}
#         response = client.post(
#             f"/plate", files=file, data={"token": "test"}
#         )
#     assert response.status_code == 404