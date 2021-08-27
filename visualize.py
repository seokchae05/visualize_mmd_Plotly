import json
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.graph_objs as go
from plotly.subplots import make_subplots


import numpy as np


import sys
import os
# from collections import OrderedDict

class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        # -------------------20210827-------
        self.dict_list = list()
        self.loss_rpn_bbox = list()
        self.loss_rpn_cls = list()
        self.s0loss_cls_objectness = list()
        self.s0loss_cls_classes = list()
        self.s0acc_objectness = list()
        self.s0acc_classes = list()
        self.s0loss_bbox = list()
        self.s0loss_mask = list()
        self.s1loss_cls_objectness = list()
        self.s1loss_cls_classes = list()
        self.s1acc_objectness = list()
        self.s1acc_classes = list()
        self.s1loss_bbox = list()
        self.s1loss_mask = list()
        self.s2loss_cls_objectness = list()
        self.s2loss_cls_classes = list()
        self.s2acc_objectness = list()
        self.s2acc_classes = list()
        self.s2loss_bbox = list()
        self.s2loss_mask = list()
        # -------------------20210827-------
        '''
        self.dict_list = list()
        self.loss_rpn_bbox = list()
        self.loss_rpn_cls = list()
        self.loss_bbox = list()
        self.loss_cls = list()
        self.loss = list()
        self.acc = list()
        '''
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
            
        json_list = ('loss_rpn_cls','loss_rpn_bbox','s0.loss_cls_objectness','s0.loss_cls_classes','s0.acc_objectness','s0.acc_classes','s0.loss_bbox','s0.loss_mask',\
            's1.loss_cls_objectness','s1.loss_cls_classes','s1.acc_objectness','s1.acc_classes','s1.loss_bbox','s1.loss_mask',\
                's2.loss_cls_objectness','s2.loss_cls_classes','s2.acc_objectness','s2.acc_classes','s2.loss_bbox','s2.loss_mask')
        
        for i in range(1, len(self.dict_list)):
            highest_iter = 700
                # ------------find key for every iter-------------------#
                
            if dict(self.dict_list[i])['mode'] == "train" and dict(self.dict_list[i])['iter'] == highest_iter:
                
                for _, jsonKey in enumerate(json_list):
                    value = dict(self.dict_list[i])[jsonKey]
                    jsonKey = jsonKey.replace(".","")
                    _tmp = getattr(self, jsonKey)
                    _tmp.append(value)
                    setattr(self, jsonKey,_tmp)
                    
                '''
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
                '''
                
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
        # ------------------clear repeated value---------------------#
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
        json_list = ('loss_rpn_cls','loss_rpn_bbox','s0loss_cls_objectness','s0loss_cls_classes','s0acc_objectness','s0acc_classes','s0loss_bbox','s0loss_mask',\
            's1loss_cls_objectness','s1loss_cls_classes','s1acc_objectness','s1acc_classes','s1loss_bbox','s1loss_mask',\
                's2loss_cls_objectness','s2loss_cls_classes','s2acc_objectness','s2acc_classes','s2loss_bbox','s2loss_mask')
        
        fig = make_subplots(rows=5, cols=5 , subplot_titles = json_list)
        for idx, jsonKey in enumerate(json_list):
            
            r = idx//5
            c = (idx % 5)
            
            fig.add_trace(
                go.Scatter(
                    x=numOfEpoch, y=getattr(self, jsonKey),name = f"{jsonKey}"
                ),
                row= r + 1, col=  c + 1
            )
            
            fig.add_trace(go.Scatter( x =[getattr(self, jsonKey).index(max(getattr(self, jsonKey))),getattr(self, jsonKey).index(min(getattr(self, jsonKey)))],  y=[max(getattr(self, jsonKey)),min(getattr(self, jsonKey))], mode = 'markers', 
                    marker=dict(color='red',size=5), text = f"max: {max(getattr(self, jsonKey))}"),row= r + 1, col= c + 1)
        '''
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.s0.loss_cls_classes,name = "s0.loss_cls_classes"
            ),
            row= 1, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.s0.loss_cls_classes.index(max(self.s0.loss_cls_classes)),self.s0.loss_cls_classes.index(min(self.s0.loss_cls_classes))],  y=[max(self.s0.loss_cls_classes),min(self.s0.loss_cls_classes)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.s0.loss_cls_classes)}"),row=  1, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.s0.acc_objectness,name = "s0.acc_objectness"
            ),
            row= 1, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.s0.acc_objectness.index(max(self.s0.acc_objectness)),self.s0.acc_objectness.index(min(self.s0.acc_objectness))],  y=[max(self.s0.acc_objectness),min(self.s0.acc_objectness)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.s0.acc_objectness)}"),row=  1, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.s0.acc_classes,name = "s0.acc_classes"
            ),
            row= 2, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.s0.acc_classes.index(max(self.s0.acc_classes)),self.s0.acc_classes.index(min(self.s0.acc_classes))],  y=[max(self.s0.acc_classes),min(self.s0.acc_classes)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.s0.acc_classes)}"),row=  2, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.s0.loss_bbox,name = "s0.loss_bbox"
            ),
            row= 2, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.s0.loss_bbox.index(max(self.s0.loss_bbox)),self.s0.loss_bbox.index(min(self.s0.loss_bbox))],  y=[max(self.s0.loss_bbox),min(self.s0.loss_bbox)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.s0.loss_bbox)}"),row=  2, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.s0.loss_mask,name = "s0.loss_mask"
            ),
            row= 2, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.s0.loss_mask.index(max(self.s0.loss_mask)),self.s0.loss_mask.index(min(self.s0.loss_mask))],  y=[max(self.s0.loss_mask),min(self.s0.loss_mask)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.s0.loss_mask)}"),row=  2, col= 3)
        '''
        fig.update_layout(height=800, width=1200,title_text="training result", template="seaborn", showlegend=False)
        
        fig.write_image(('output/' + sys.argv[1][5:] + 'train_result.png'))
        fig.show()
        
        
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
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP_s)}"), row=  1, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x= numOfEpoch, y=self.bbox_mAP_m,name = "bbox_mAP_m"
            ),
            row= 1, col= 2
        )
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_m.index(max(self.bbox_mAP_m)),self.bbox_mAP_m.index(min(self.bbox_mAP_m))],  y=[max(self.bbox_mAP_m),min(self.bbox_mAP_m)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP_m)}"),row=  1, col= 2)
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_l,name = "bbox_mAP_l"
            ),
            row= 1, col= 3
        )
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_l.index(max(self.bbox_mAP_l)),self.bbox_mAP_l.index(min(self.bbox_mAP_l))],  y=[max(self.bbox_mAP_l),min(self.bbox_mAP_l)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP_l)}"),row=  1, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP,name = "bbox_mAP"
            ),
            row= 1, col= 4
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP.index(max(self.bbox_mAP)),self.bbox_mAP.index(min(self.bbox_mAP))],  y=[max(self.bbox_mAP),min(self.bbox_mAP)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP)}"),row=  1, col= 4)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_50,name = "bbox_mAP_50"
            ),
            row= 2, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_50.index(max(self.bbox_mAP_50)),self.bbox_mAP_50.index(min(self.bbox_mAP_50))],  y=[max(self.bbox_mAP_50),min(self.bbox_mAP_50)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP_50)}"),row=  2, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.bbox_mAP_75,name = "bbox_mAP_75"
            ),
            row= 2, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.bbox_mAP_75.index(max(self.bbox_mAP_75)),self.bbox_mAP_75.index(min(self.bbox_mAP_75))],  y=[max(self.bbox_mAP_75),min(self.bbox_mAP_75)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.bbox_mAP_75)}"),row=  2, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_s,name = "segm_mAP_s"
            ),
            row= 3, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_s.index(max(self.segm_mAP_s)),self.segm_mAP_s.index(min(self.segm_mAP_s))],  y=[max(self.segm_mAP_s),min(self.segm_mAP_s)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.segm_mAP_s)}"),row=  3, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_m,name = "segm_mAP_m"
            ),
            row= 3, col= 2
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_m.index(max(self.segm_mAP_m)),self.segm_mAP_m.index(min(self.segm_mAP_m))],  y=[max(self.segm_mAP_m),min(self.segm_mAP_m)], mode = 'markers', 
                marker=dict(color='red',size=5),text = f"max: {max(self.segm_mAP_m)}"),row=  3, col= 2)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_l,name = "segm_mAP_l"
            ),
            row= 3, col= 3
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_l.index(max(self.segm_mAP_l)),self.segm_mAP_l.index(min(self.segm_mAP_l))],  y=[max(self.segm_mAP_l),min(self.segm_mAP_l)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.segm_mAP_l)}"),row=  3, col= 3)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP,name = "segm_mAP"
            ),
            row= 3, col= 4
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP.index(max(self.segm_mAP)),self.segm_mAP.index(min(self.segm_mAP))],  y=[max(self.segm_mAP),min(self.segm_mAP)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.segm_mAP)}"),row=  3, col= 4)
        
        fig.add_trace(
            go.Scatter(
                y=self.segm_mAP_50,name = "segm_mAP_50"
            ),
            row= 4, col= 1
        )
        
        fig.add_trace(go.Scatter( x =[self.segm_mAP_50.index(max(self.segm_mAP_50)),self.segm_mAP_50.index(min(self.segm_mAP_50))],  y=[max(self.segm_mAP_50),min(self.segm_mAP_50)], mode = 'markers', 
                marker=dict(color='red',size=5), text = f"max: {max(self.segm_mAP_50)}"),row=  4, col= 1)
        
        fig.add_trace(
            go.Scatter(
                x=numOfEpoch, y=self.segm_mAP_75,name = "segm_mAP_75"
            ),
            row= 4, col= 2
        )
        fig.add_trace(go.Scatter( x =[self.segm_mAP_75.index(max(self.segm_mAP_75)),self.segm_mAP_75.index(min(self.segm_mAP_75))],  y=[max(self.segm_mAP_75),min(self.segm_mAP_75)], mode = 'markers', 
                marker=dict(color='red',size=5),text = f"max: {max(self.segm_mAP_75)}") ,row=  4, col= 2)
        
        
        fig.update_layout(height=800, width=1200,title_text= "validation result", template="seaborn",showlegend= False)
        # fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        fig.update_traces(textposition='top center')
        fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        
        fig.show()


if __name__ == '__main__':
    x = visualize_mmdetection(sys.argv[1])
    x.load_data()
    x.show_chart_train()
    x.show_chart_val()