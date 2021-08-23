import json
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.graph_objs as go
from plotly.subplots import make_subplots





import sys
import os
from collections import OrderedDict

class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)

        self.dict_list = list()
        self.loss_rpn_bbox = list()
        self.loss_rpn_cls = list()
        self.loss_bbox = list()
        self.loss_cls = list()
        self.loss = list()
        self.acc = list()
        # -------------------val ------
        self.bbox_mAP = list()
        self.bbox_mAP_50 = list()
        self.bbox_mAP_75 = list()
        self.bbox_mAP_s = list()
        self.bbox_mAP_m = list()
        self.bbox_mAP_l = list()
        self.segm_mAP = list()
        self.segm_mAP_50 = list()
        self.segm_mAP_75 = list()
        self.segm_mAP_s = list()
        self.segm_mAP_m = list()
        self.segm_mAP_l = list()


    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            self.dict_list.append(info)

        for i in range(1, len(self.dict_list)):
                # ------------find key for every iter-------------------#
            if dict(self.dict_list[i])['mode'] == "train" and dict(self.dict_list[i])['iter'] == 700:
                    
            
                    
                loss_rpn_cls_value = dict(self.dict_list[i])['loss_rpn_cls']
                
                loss_rpn_bbox_value = dict(self.dict_list[i])['loss_rpn_bbox']
                loss_bbox_value = dict(self.dict_list[i])['loss_bbox']
                loss_cls_value = dict(self.dict_list[i])['loss_cls']
                loss_value = dict(self.dict_list[i])['loss']
                acc_value = dict(self.dict_list[i])['acc']
                    # -------------list append------------------------------#

                self.loss_rpn_cls.append(loss_rpn_cls_value)
                
                
                self.loss_rpn_bbox.append(loss_rpn_bbox_value)
                self.loss_bbox.append(loss_bbox_value)
                self.loss_cls.append(loss_cls_value)
                self.loss.append(loss_value)
                self.acc.append(acc_value)
                
                
            elif dict(self.dict_list[i])['mode'] == "val" and self.dict_list[i]['epoch'] != 1 :
                    
                bbox_mAP_value = dict(self.dict_list[i])['bbox_mAP']
                bbox_mAP_50_value = dict(self.dict_list[i])['bbox_mAP_50']
                bbox_mAP_75_value = dict(self.dict_list[i])['bbox_mAP_75']
                bbox_mAP_s_value = dict(self.dict_list[i])['bbox_mAP_s']
                bbox_mAP_m_value = dict(self.dict_list[i])['bbox_mAP_m']
                bbox_mAP_l_value = dict(self.dict_list[i])['bbox_mAP_l']
                segm_mAP_value = dict(self.dict_list[i])['segm_mAP']
                segm_mAP_50_value = dict(self.dict_list[i])['segm_mAP_50']
                segm_mAP_75_value = dict(self.dict_list[i])['segm_mAP_75']
                segm_mAP_s_value = dict(self.dict_list[i])['segm_mAP_s']
                segm_mAP_m_value = dict(self.dict_list[i])['segm_mAP_m']
                segm_mAP_l_value = dict(self.dict_list[i])['segm_mAP_l']
                    
                    
                self.bbox_mAP.append(bbox_mAP_value)
                self.bbox_mAP_50.append(bbox_mAP_50_value)
                self.bbox_mAP_75.append(bbox_mAP_75_value)
                self.bbox_mAP_s.append(bbox_mAP_s_value)
                self.bbox_mAP_m.append(bbox_mAP_m_value)
                self.bbox_mAP_l.append(bbox_mAP_l_value)
                
                self.segm_mAP.append(segm_mAP_value)
                self.segm_mAP_50.append(segm_mAP_50_value)
                self.segm_mAP_75.append(segm_mAP_75_value)
                self.segm_mAP_s.append(segm_mAP_s_value)
                self.segm_mAP_m.append(segm_mAP_m_value)
                self.segm_mAP_l.append(segm_mAP_l_value)

            else:
                continue
            # to avoid dict no have value 
                # -------------clear repeated value---------------------#
        # self.loss_rpn_cls = list(OrderedDict.fromkeys(self.loss_rpn_cls))
        # self.loss_rpn_bbox = list(OrderedDict.fromkeys(self.loss_rpn_bbox))
        # self.loss_bbox = list(OrderedDict.fromkeys(self.loss_bbox))
        # self.loss_cls = list(OrderedDict.fromkeys(self.loss_cls))
        # self.loss = list(OrderedDict.fromkeys(self.loss))
        # self.acc = list(OrderedDict.fromkeys(self.acc))
        
        # self.bbox_mAP = list(OrderedDict.fromkeys(self.bbox_mAP))
        # self.bbox_mAP_50 = list(OrderedDict.fromkeys(self.bbox_mAP_50))
        # self.bbox_mAP_75 = list(OrderedDict.fromkeys(self.bbox_mAP_75))
        # self.bbox_mAP_s = list(OrderedDict.fromkeys(self.bbox_mAP_s))
        # self.bbox_mAP_m = list(OrderedDict.fromkeys(self.bbox_mAP_m))
        # self.bbox_mAP_l = list(OrderedDict.fromkeys(self.bbox_mAP_l))
        
        # self.segm_mAP = list(OrderedDict.fromkeys(self.segm_mAP))
        # self.segm_mAP_50 = list(OrderedDict.fromkeys(self.segm_mAP_50))
        # self.segm_mAP_75 = list(OrderedDict.fromkeys(self.segm_mAP_75))
        # self.segm_mAP_s = list(OrderedDict.fromkeys(self.segm_mAP_s))
        # self.segm_mAP_m = list(OrderedDict.fromkeys(self.segm_mAP_m))
        # self.segm_mAP_l = list(OrderedDict.fromkeys(self.segm_mAP_l))

    def show_chart_train(self):
        numOfEpoch = list(range(1,201))
        fig = make_subplots(rows=2, cols=3 , subplot_titles =("loss_rpn_cls","loss_rpn_bbox", "loss_cls","loss_bbox", "total loss", "accuracy" ))

        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.loss_rpn_cls,name = "loss_rpn_cls"
            ),
            row= 1, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.loss_rpn_cls.index(max(self.loss_rpn_cls)),self.loss_rpn_cls.index(min(self.loss_rpn_cls))],  y=[max(self.loss_rpn_cls),min(self.loss_rpn_cls)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.loss_rpn_cls)}"),row=  1, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.loss_rpn_bbox,name = "loss_rpn_bbox"
            ),
            row= 1, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.loss_rpn_bbox.index(max(self.loss_rpn_bbox)),self.loss_rpn_bbox.index(min(self.loss_rpn_bbox))],  y=[max(self.loss_rpn_bbox),min(self.loss_rpn_bbox)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.loss_rpn_bbox)}"),row=  1, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.loss_cls,name = "loss_cls"
            ),
            row= 1, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.loss_cls.index(max(self.loss_cls)),self.loss_cls.index(min(self.loss_cls))],  y=[max(self.loss_cls),min(self.loss_cls)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.loss_cls)}"),row=  1, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.loss_bbox,name = "loss_bbox"
            ),
            row= 2, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.loss_bbox.index(max(self.loss_bbox)),self.loss_bbox.index(min(self.loss_bbox))],  y=[max(self.loss_bbox),min(self.loss_bbox)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.loss_bbox)}"),row=  2, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.loss,name = "total loss"
            ),
            row= 2, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.loss.index(max(self.loss)),self.loss.index(min(self.loss))],  y=[max(self.loss),min(self.loss)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.loss)}"),row=  2, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.acc,name = "accuracy"
            ),
            row= 2, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.acc.index(max(self.acc)),self.acc.index(min(self.acc))],  y=[max(self.acc),min(self.acc)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.acc)}"),row=  2, col= 3)
        
        fig.update_layout(height=800, width=1200,title_text="training result", template="seaborn")
        
        fig.write_image(('output/' + sys.argv[1][5:] + 'train_result.png'))
        
        
    def show_chart_val(self):
        
        
        numOfEpoch = list(range(1,200))
        
        fig = make_subplots(rows=4, cols=4 ,subplot_titles = ("bbox_mAP_s","bbox_mAP_m","bbox_mAP_l",'bbox_mAP','bbox_mAP_50','bbox_mAP_75','','',"segm_mAP_s","segm_mAP_m","segm_mAP_l",'segm_mAP','segm_mAP_50','segm_mAP_75'))
        
        

        fig.add_trace(
            go.Scatter(
                x= numOfEpoch, y=self.bbox_mAP_s, name = "bbox_mAP_s"
            ),
            row= 1, col= 1
        )
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_s.index(max(self.bbox_mAP_s)),self.bbox_mAP_s.index(min(self.bbox_mAP_s))],  y=[max(self.bbox_mAP_s),min(self.bbox_mAP_s)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP_s)}"),row=  1, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x= numOfEpoch, y=self.bbox_mAP_m,name = "bbox_mAP_m"
            ),
            row= 1, col= 2
        )
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_m.index(max(self.bbox_mAP_m)),self.bbox_mAP_m.index(min(self.bbox_mAP_m))],  y=[max(self.bbox_mAP_m),min(self.bbox_mAP_m)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP_m)}"),row=  1, col= 2)
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_l,name = "bbox_mAP_l"
            ),
            row= 1, col= 3
        )
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_l.index(max(self.bbox_mAP_l)),self.bbox_mAP_l.index(min(self.bbox_mAP_l))],  y=[max(self.bbox_mAP_l),min(self.bbox_mAP_l)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP_l)}"),row=  1, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP,name = "bbox_mAP"
            ),
            row= 1, col= 4
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP.index(max(self.bbox_mAP)),self.bbox_mAP.index(min(self.bbox_mAP))],  y=[max(self.bbox_mAP),min(self.bbox_mAP)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP)}"),row=  1, col= 4)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_50,name = "bbox_mAP_50"
            ),
            row= 2, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_50.index(max(self.bbox_mAP_50)),self.bbox_mAP_50.index(min(self.bbox_mAP_50))],  y=[max(self.bbox_mAP_50),min(self.bbox_mAP_50)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP_50)}"),row=  2, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_75,name = "bbox_mAP_75"
            ),
            row= 2, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_75.index(max(self.bbox_mAP_75)),self.bbox_mAP_75.index(min(self.bbox_mAP_75))],  y=[max(self.bbox_mAP_75),min(self.bbox_mAP_75)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.bbox_mAP_75)}"),row=  2, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_s,name = "segm_mAP_s"
            ),
            row= 3, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_s.index(max(self.segm_mAP_s)),self.segm_mAP_s.index(min(self.segm_mAP_s))],  y=[max(self.segm_mAP_s),min(self.segm_mAP_s)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.segm_mAP_s)}"),row=  3, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_m,name = "segm_mAP_m"
            ),
            row= 3, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_m.index(max(self.segm_mAP_m)),self.segm_mAP_m.index(min(self.segm_mAP_m))],  y=[max(self.segm_mAP_m),min(self.segm_mAP_m)], mode = 'markers', 
                marker=dict(color='red',size=5),name = f"max: {max(self.segm_mAP_m)}"),row=  3, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_l,name = "segm_mAP_l"
            ),
            row= 3, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_l.index(max(self.segm_mAP_l)),self.segm_mAP_l.index(min(self.segm_mAP_l))],  y=[max(self.segm_mAP_l),min(self.segm_mAP_l)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.segm_mAP_l)}"),row=  3, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP,name = "segm_mAP"
            ),
            row= 3, col= 4
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP.index(max(self.segm_mAP)),self.segm_mAP.index(min(self.segm_mAP))],  y=[max(self.segm_mAP),min(self.segm_mAP)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.segm_mAP)}"),row=  3, col= 4)
        
        fig.add_trace(
            go.Scatter(
                y=self.segm_mAP_50,name = "segm_mAP_50"
            ),
            row= 4, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_50.index(max(self.segm_mAP_50)),self.segm_mAP_50.index(min(self.segm_mAP_50))],  y=[max(self.segm_mAP_50),min(self.segm_mAP_50)], mode = 'markers', 
                marker=dict(color='red',size=5), name = f"max: {max(self.segm_mAP_50)}"),row=  4, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_75,name = "segm_mAP_75"
            ),
            row= 4, col= 2
        )
        fig.add_trace(go.Scatter( x =[self.segm_mAP_75.index(max(self.segm_mAP_75)),self.segm_mAP_75.index(min(self.segm_mAP_75))],  y=[max(self.segm_mAP_75),min(self.segm_mAP_75)], mode = 'markers', 
                marker=dict(color='red',size=5),name = f"max: {max(self.segm_mAP_75)}") ,row=  4, col= 2)
        
        
        fig.update_layout(height=800, width=1200,title_text= "validation result", template="seaborn")
        # fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        fig.update_traces(textposition='top center')
        fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        
        


if __name__ == '__main__':
    x = visualize_mmdetection(sys.argv[1])
    x.load_data()
    x.show_chart_train()
    x.show_chart_val()