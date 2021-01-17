# from app.tests.utils.utils import random_lower_string
from fastapi.testclient import TestClient
# from sqlalchemy.orm import Session

# from app.core.config import settings
# from app.tests.utils.image import create_random_business_image, create_random_image, create_random_activity_image
# from app.tests.utils.business import create_random_business


def test_upload_image(client: TestClient) -> None:
    with open("app/tests/vuplate.jpg", "rb") as f:
        file = {"image": ("vuplate.jpg", f, "image/jpeg")}
        response = client.post(
            f"/api/v1/detect/vehicles", files=file, data={"token": "test"}
        )
        assert response.status_code == 201
        content = response.json()


def test_upload_image_read_status(client: TestClient) -> None:
    with open("app/tests/vuplate.jpg", "rb") as f:
        file = {"image": ("vuplate.jpg", f, "image/jpeg")}
        response = client.post(
            f"/api/v1/detect/vehicles", files=file, data={"token": "test"}
        )
    assert response.status_code == 201
    content = response.json() 
    taskId = content["taskId"]
    response = client.get(
        f"/api/v1/detect/vehicles/{taskId}"
    )
    assert response.status_code == 200


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


def test_colour_detection(client: TestClient) -> None:
    with open("app/tests/vuplate.jpg", "rb") as f:
        file = {"image": ("vuplate.jpg", f, "image/jpeg")}
        response = client.post(
            f"/api/v1/detect/colours", files=file, data={"token": "test"}
        )
        assert response.status_code == 201
        content = response.json() 
        taskId = content["taskId"]
        response = client.get(
            f"/api/v1/detect/vehicles/{taskId}"
        )
        assert response.status_code == 200