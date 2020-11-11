# Transport ANPR API

I plan this repo to house the code related to deep learning and return simple responses to given images. Therefore the deep learning models and code will not clutter my main API which is responsible for users, vehicles, votes, comments, images, etc. 

I want this project to be well organized and it might also keep the materials for training the models and not just using the models so I can expand upon them when I get more local data to fine-tune the models.

## Personal Thoughts

I've seen a few different ways to detect the number plate including solely using a tranined YOLO model that depended upon the plates being regular which is not an option for my use case. I've also seen some implementations that do not affine the plate which caused the OCR to fail. To affine the plate seems like an obvious step to take; so was there a reason for that but I don't know yet?

It seems wpod-net is newer better implementation of number plate detection since I've first looked into this project a couple years ago. OpenALPR does not seem to be coming up high in any the search results I look for any longer even though it is still maintained/developed. Hard to find any comparisons of both at the moment until I learn more about both and see if I'm misunderstanding their roles/objectives.

## Reference Materials

https://github.com/shanesoh/deploy-ml-fastapi-redis-docker

https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

Get plates from video script and traning scripts for plates/characters
https://github.com/stevefielding/keras-anpr
https://github.com/TheophileBuy/LicensePlateRecognition  - YOLO implementation

ALPR-Unconstrained

http://www.inf.ufrgs.br/~smsilva/alpr-unconstrained/

Training scripts and annotation tool
https://github.com/sergiomsilva/alpr-unconstrained  - wpod-net

https://github.com/wechao18/Efficient-alpr-unconstrained  - ??? efficient ???

OpenALPR

https://github.com/openalpr/openalpr
https://github.com/openalpr/train-detector
https://github.com/openalpr/train-ocr


Redis w/ FastAPI inspiration

https://github.com/leonh/redis-streams-fastapi-chat/blob/master/chat.py