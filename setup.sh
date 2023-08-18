# https://neuraspike.com/blog/training-emotion-detection-system-pytorch/


# Display Results
python3 train.py --model output/model.pth --plot output/plot.png


# Display OpenCV Flip Result
python emotion_detection.py -i video/novak_djokovic.mp4 --model output/model.pth --prototxt model/deploy.prototxt --caffemodel model/res10_300x300_ssd_iter_140000_fp16.caffemodel