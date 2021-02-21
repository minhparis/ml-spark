Gradient descent optimization in Spark

Big data solutions focus mainly on the Extraction and Transformation aspect of processing. The MapReduce model allows us to easily implement information extractions, but many constraints and limitations appear when designing data algorithms.
For example, the iterative algorithms commonly used in machine learning are difficult to integrate into MapReduce models: the high level of data interaction requires complex management and synchronization at different stages of the analysis.
In this project we will try to face this difficulty and apply in a typical use case in machine learning: the design of an SGD model. Our goal is to demonstrate the adaptability and elegance of implementation using the distributed computing framework Spark.
