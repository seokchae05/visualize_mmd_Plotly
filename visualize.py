import json
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.graph_objs as go
from plotly.subplots import make_subplots


import numpy as np


import sys
import os
# from collections import OrderedDict

"""
    1. initialize json key -> List
    2. set json_listT and V that can harmony with your json file
    3. setting col,row to make viusalizing 

"""


class visualize_mmdetection():
    def __init__(self, path):
        self.log = open(path)
        # -------------------20210827-------
        self.dict_list = []
        self.loss_rpn_bbox = []
        self.loss_rpn_cls = []
        self.s0loss_cls_objectness =[]
        self.s0loss_cls_classes =[]
        self.s0acc_objectness = []
        self.s0acc_classes =[]
        self.s0loss_bbox = []
        self.s0loss_mask =[]
        self.s1loss_cls_objectness =[]
        self.s1acc_objectness = []
        self.s1acc_classes = []
        self.s1loss_bbox = []
        self.s1loss_mask = []
        self.s2loss_cls_objectness = []
        self.s2loss_cls_classes = []
        self.s2acc_objectness = []
        self.s2acc_classes = []
        self.s2loss_bbox = []
        self.s2loss_mask = []
        # -------------------20210827-------
        '''
        self.dict_list = []
        self.loss_rpn_bbox = []
        self.loss_rpn_cls = []
        self.loss_bbox = []
        self.loss_cls = []
        self.loss = []
        self.acc = []
        '''
        # -------------------val ------
        self.bbox_mAP = []
        self.bbox_mAP_50 = []
        self.bbox_mAP_75 = []
        self.bbox_mAP_s = []
        self.bbox_mAP_m = []
        self.bbox_mAP_l = []
        self.segm_mAP = []
        self.segm_mAP_50 = []
        self.segm_mAP_75 = []
        self.segm_mAP_s = []
        self.segm_mAP_m = []
        self.segm_mAP_l = []


    def load_data(self):
        for line in self.log:
            info = json.loads(line)
            self.dict_list.append(info)
        #--------------------------------setJSonList--------------------------#
        json_listT = ('s0.loss_cls_objectness','s0.loss_cls_classes','s0.acc_objectness','s0.acc_classes','s0.loss_bbox','s0.loss_mask',\
            's1.loss_cls_objectness','s1.loss_cls_classes','s1.acc_objectness','s1.acc_classes','s1.loss_bbox','s1.loss_mask',\
                's2.loss_cls_objectness','s2.loss_cls_classes','s2.acc_objectness','s2.acc_classes','s2.loss_bbox','s2.loss_mask','loss_rpn_cls','loss_rpn_bbox')
        
        json_listV = ('bbox_mAP',"bbox_mAP_50","bbox_mAP_75","bbox_mAP_s", "bbox_mAP_s", "bbox_mAP_m","bbox_mAP_l",'segm_mAP',"segm_mAP_50","segm_mAP_75","segm_mAP_s","segm_mAP_m","segm_mAP_l")
        
        
        for i in range(1, len(self.dict_list)):
            highest_iter = 1400
                # ------------find key for every iter-------------------#
                
            if dict(self.dict_list[i])['mode'] == "train" and dict(self.dict_list[i])['iter'] == highest_iter:
                
                for _, jsonValue in enumerate(json_listT):
                    value = dict(self.dict_list[i])[jsonValue]
                    jsonValue = jsonValue.replace(".","")
                    _tmp = getattr(self, jsonValue)
                    _tmp.append(value)
                    setattr(self, jsonValue,_tmp)
                    
                
                
            elif dict(self.dict_list[i])['mode'] == "val" and self.dict_list[i]['epoch'] != 1 :
                
                for _, jsonValue in enumerate(json_listV):
                    value = dict(self.dict_list[i])[jsonValue]
                    jsonValue = jsonValue.replace(".","")
                    _tmp = getattr(self, jsonValue)
                    _tmp.append(value)
                    setattr(self, jsonValue,_tmp)

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
        json_listT = ('s0loss_cls_objectness','s0loss_cls_classes','s0acc_objectness','s0acc_classes','s0loss_bbox','s0loss_mask',\
            's1loss_cls_objectness','s1loss_cls_classes','s1acc_objectness','s1acc_classes','s1loss_bbox','s1loss_mask',\
                's2loss_cls_objectness','s2loss_cls_classes','s2acc_objectness','s2acc_classes','s2loss_bbox','s2loss_mask','loss_rpn_cls','loss_rpn_bbox')
        
        fig = make_subplots(rows=6, cols=6 , subplot_titles = json_listT)
        for idx, jsonValue in enumerate(json_listT):
            
            r = idx//6
            c = (idx % 6)
            
            fig.add_trace(
                go.Scatter(
                    x=numOfEpoch, y=getattr(self, jsonValue),name = f"{jsonValue}"
                ),
                row= r + 1, col=  c + 1
            )
            
            val = "loss" in jsonValue   #find specific letter -> bool
            
            if val == False:
                fig.add_trace(go.Scatter( x =[getattr(self, jsonValue).index(max(getattr(self, jsonValue)))],  y=[max(getattr(self, jsonValue))], mode = 'markers+text', 
                    marker=dict(color='red',size=3), textposition="bottom left",textfont = dict( size = 9),text = f"{max(getattr(self, jsonValue))}"),row= r + 1, col= c + 1)
            
                fig.add_trace(go.Scatter( x =[getattr(self, jsonValue).index(min(getattr(self, jsonValue)))],  y=[min(getattr(self, jsonValue))], mode = 'markers+text', 
                    marker=dict(color='red',size=3), textposition="top right",textfont = dict( size = 9), text = f"{min(getattr(self, jsonValue))}"),row= r + 1, col= c + 1)
            elif val == True:
                fig.add_trace(go.Scatter( x =[getattr(self, jsonValue).index(max(getattr(self, jsonValue)))],  y=[max(getattr(self, jsonValue))], mode = 'markers+text', 
                    marker=dict(color='red',size=3), textposition="bottom right",textfont = dict( size = 9),text = f"{max(getattr(self, jsonValue))}"),row= r + 1, col= c + 1)
            
                fig.add_trace(go.Scatter( x =[getattr(self, jsonValue).index(min(getattr(self, jsonValue)))],  y=[min(getattr(self, jsonValue))], mode = 'markers+text', 
                    marker=dict(color='red',size=3), textposition="top left",textfont = dict( size = 9), text = f"{min(getattr(self, jsonValue))}"),row= r + 1, col= c + 1)
        
        
        fig.update_layout(height=800, width=1200,title_text="training result", template="seaborn", showlegend=False)
        
        fig.write_image(('output/' + sys.argv[1][5:] + 'train_result.png'))
        fig.show()
        
        
    def show_chart_val(self):
        numOfEpoch = list(range(1,200))
        json_listV = ("bbox_mAP_s","bbox_mAP_m","bbox_mAP_l",'bbox_mAP','bbox_mAP_50','bbox_mAP_75','','',"segm_mAP_s","segm_mAP_m","segm_mAP_l",'segm_mAP','segm_mAP_50','segm_mAP_75')
        
        fig = make_subplots(rows=4, cols=4 ,subplot_titles = json_listV)
        
        
        for idx, jsonValue in enumerate(json_listV):
            try:
                r = idx//4
                c = (idx % 4)
                
                fig.add_trace(
                    go.Scatter(
                    x=numOfEpoch, y=getattr(self, jsonValue),name = f"{jsonValue}"
                    ),
                row= r + 1, col= c + 1
                )
                fig.add_trace(go.Scatter( x =[getattr(self, jsonValue).index(max(getattr(self, jsonValue)))],  y=[max(getattr(self, jsonValue))], mode = 'markers+text', 
                    marker=dict(color='red',size=3), textposition="bottom left",textfont = dict( size = 9),text = f"{max(getattr(self, jsonValue))}"),row= r + 1, col= c + 1)
            except:
                continue
        
        fig.update_layout(height=800, width=1200,title_text= "validation result", template="seaborn",)
        # fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        fig.update_traces(textposition='top center')
        fig.write_image(('output/' + sys.argv[1][5:] + 'val_result.png'))
        
        fig.show()


if __name__ == '__main__':
    x = visualize_mmdetection(sys.argv[1])
    x.load_data()
    x.show_chart_train()
    x.show_chart_val()