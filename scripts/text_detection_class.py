import tensorflow as tf
from tensorflow.python.platform import gfile    # used to read the frozen model file 

class TextDetector(object):
    def __init__(self):
        # create tensorflow session that can use any available devices for operations
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.session = tf.compat.v1.Session(config=config)

        # open the pretrained tensorflow model in read-binary mode. 
        with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()             # creates graph buffer object 
            graph_def.ParseFromString(f.read())             # reads file and parses into graphdef       
            self.session.graph.as_default()                 # sets session graph as default
            tf.import_graph_def(graph_def, name='')         # imports graph_def operations into current graph

        # retrieve tensorflow tensors from current graph
        self.input_img = self.session.graph.get_tensor_by_name('Placeholder:0')                   # input of graph
        self.output_cls_prob = self.session.graph.get_tensor_by_name('Reshape_2:0')               # output of graph
        self.output_box_pred = self.session.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0') # output of graph
    
    def extract_text_info(self, blobs):
        # coputes classifaction probabilities and box predictions
        cls_prob, box_pred = self.session.run([self.output_cls_prob, self.output_box_pred], feed_dict={self.input_img: blobs['data']})
        return cls_prob, box_pred