All the code we need is in ricoh.py, you can ignore everything else.

There's a comment that demonstrates where we have an estimate of the position and orientation which can be sent over the network. If that value is None, it's because there are no readings in the camera image.