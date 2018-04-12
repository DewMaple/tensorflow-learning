# Tensorflow learning
A sample repository of my tensorflow learning


# Note 1

**Constant values are `stored` in the graph definition**

**Sessions `allocate memory` to store variable values**

# Note 2

**Feed values into placeholders with a dictionary (feed_dict)**

**Easy to use but `poor` performance**

# Note 3

**Separate the assembling of graph and executing ops**

**Use Python attribute to ensure a function is `only` loaded the `first` time it’s called**


# Note 4 Should we always use tf.data?

**For prototyping, feed dict can be faster and easier to write (pythonic)**

**tf.data is tricky to use when you have complicated preprocessing or multiple data sources**

**NLP data is normally just a sequence of integers. In this case, transferring the data over to GPU is pretty quick, so the speedup of tf.data isn't that large**
