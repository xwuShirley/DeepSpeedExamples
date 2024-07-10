import json
with open('/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/31k_lvis/lvis_xiaoxiao_28903.json') as f:
    xiaoxiao1=json.load(f)
 
# path1 = ['/blob/data/obj-render-13views-xiaoxiao/']      
with open('/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/50k_mask01_06_std011/xiaoxiao_mask01-06_stdL011_2252.json') as f:
    xiaoxiao2=json.load(f)
  
with open(f'/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/xiaoxiao_{len(xiaoxiao1)+len(xiaoxiao2)}.json', 'w') as f:
    json.dump(xiaoxiao1+xiaoxiao2,f)
    
    
    
with open('/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/31k_lvis/lvis_ziyuan_10299.json') as f:
    ziyuan1=json.load(f)
 
# path1 = ['/blob/data/obj-render-13views-xiaoxiao/']      
with open('/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/50k_mask01_06_std011/ziyuan_mask01-06_stdL011_48773.json') as f:
    ziyuan2=json.load(f)
  
with open(f'/vc_data/users/xwu/Model3d/render-data/mvdiffusion/wonder3d-fullcode/data_lists/final_train_data/ziyuan_{len(ziyuan1)+len(ziyuan2)}.json', 'w') as f:
    json.dump(ziyuan2+ziyuan1,f)
