from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "187bca90-698b-11ed-8069-d91c83abdae3d786d745-e308-4e96-a2f0-960d9da2d369"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("sneeze1.jpeg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))