from pathlib import Path
import json, csv, shutil, zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("aquaculture-ai-15")
if ROOT.exists(): shutil.rmtree(ROOT)

for p in ["data/cv/images/train", "data/cv/images/val", "data/cv/images/test",
          "data/cv/labels/train", "data/cv/labels/val", "data/cv/labels/test",
          "notebooks/module_1", "notebooks/module_2", "notebooks/module_3",
          "notebooks/module_4", "notebooks/module_5", "assessments", "docs", "scripts"]:
    (ROOT/p).mkdir(parents=True, exist_ok=True)

rng=np.random.default_rng(2607)
n=2600
t=pd.date_range("2025-01-01", periods=n, freq="h")
temp=16+3*np.sin(np.arange(n)*2*np.pi/(24*30))+rng.normal(0,.5,n)
sal=2.5+rng.normal(0,.2,n)
ph=7.4+rng.normal(0,.12,n)
amm=np.clip(.035+.012*np.sin(np.arange(n)*2*np.pi/(24*7))+rng.normal(0,.008,n),0,None)
o2=11.5-.27*temp-.08*sal-.8*amm+rng.normal(0,.25,n)
sensor=pd.DataFrame(dict(timestamp=t,temperature_C=temp,salinity_ppt=sal,dissolved_oxygen_mgL=o2,pH=ph,ammonia_mgL=amm))
for c in sensor.columns[1:]: sensor.loc[rng.choice(n,18,replace=False),c]=np.nan
sensor.to_csv(ROOT/"data/ras_sensor_log.csv",index=False)

n=600
species=rng.choice(["Rainbow trout","Siberian sturgeon"],n)
age=rng.integers(8,80,n); wt=np.where(species=="Rainbow trout",8.5*age**1.25,6.2*age**1.35)+rng.normal(0,45,n)
growth=pd.DataFrame({"species":species,"age_weeks":age,"mean_water_temp_C":rng.normal(17,2,n),"daily_feed_ration_g":np.clip(wt*.022+rng.normal(0,3,n),1,None),"length_cm":np.clip(wt,1,None)**.33*3.1+rng.normal(0,1,n),"weight_g":np.clip(wt,10,None)})
growth.to_csv(ROOT/"data/fish_growth_dataset.csv",index=False)

n=800; do=rng.normal(7,1.2,n); tt=rng.normal(18,2,n); feed=rng.beta(5,2,n); turb=rng.gamma(2,3,n)
logit=1.4*(do-6)-.18*(tt-19)+2*(feed-.5)-.08*turb+rng.normal(0,.6,n)
risk=(logit<-.5).astype(int)
pd.DataFrame({"dissolved_oxygen_mgL":do,"temperature_C":tt,"feeding_activity_index":feed,"turbidity_NTU":turb,"risk":risk}).to_csv(ROOT/"data/sturgeon_risk_classification.csv",index=False)

n=1440;t=pd.date_range("2025-05-01",periods=n,freq="h"); x=np.arange(n)
hydro=pd.DataFrame({"timestamp":t,"temperature_C":18+2*np.sin(2*np.pi*x/24)+rng.normal(0,.25,n),"dissolved_oxygen_mgL":7.8-1.2*np.sin(2*np.pi*x/24)+rng.normal(0,.18,n),"pH":7.4+.12*np.sin(2*np.pi*x/(24*5))+rng.normal(0,.04,n),"ammonia_mgL":np.clip(.04+.01*np.sin(2*np.pi*x/(24*3))+rng.normal(0,.005,n),0,None)})
hydro.to_csv(ROOT/"data/hydrochem_timeseries.csv",index=False)

ann=[]
for split,count in [("train",40),("val",10),("test",10)]:
  for i in range(count):
    W=H=416; im=Image.new("RGB",(W,H),(25,90+rng.integers(0,30),115+rng.integers(0,25))); d=ImageDraw.Draw(im)
    k=int(rng.integers(4,10)); rows=[]
    for obj in range(k):
      cx=int(rng.integers(35,W-35)); cy=int(rng.integers(25,H-25)); bw=int(rng.integers(28,58)); bh=int(rng.integers(10,22))
      box=(cx-bw//2,cy-bh//2,cx+bw//2,cy+bh//2); d.ellipse(box,fill=(180+rng.integers(0,50),150+rng.integers(0,50),70+rng.integers(0,40)))
      rows.append(f"0 {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}")
      ann.append([split,f"{split}_{i:03}.png",obj,*box])
    im.save(ROOT/f"data/cv/images/{split}/{split}_{i:03}.png")
    (ROOT/f"data/cv/labels/{split}/{split}_{i:03}.txt").write_text("\n".join(rows),encoding="utf8")
with open(ROOT/"data/cv/annotations.csv","w",newline="",encoding="utf8") as f:
  w=csv.writer(f);w.writerow(["split","image","object_id","x1","y1","x2","y2"]);w.writerows(ann)
(ROOT/"data/cv/fry_dataset.yaml").write_text("path: ../data/cv\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fry\n",encoding="utf8")

def nb(title, objectives, cells):
  C=[{"cell_type":"markdown","metadata":{},"source":[f"# {title}\n",* [f"- {x}\n" for x in objectives]]}]
  for kind,src in cells:
    C.append({"cell_type":kind,"metadata":{},"source":[line+"\n" for line in src.strip().splitlines()], **({"execution_count":None,"outputs":[]} if kind=="code" else {})})
  return {"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}

P="../../data/"
works=[
(1,"Профилирование сенсорных данных",["Загрузить данные","Выявить пропуски и аномалии"],f'''import pandas as pd\ndf=pd.read_csv("{P}ras_sensor_log.csv",parse_dates=["timestamp"])\nprint(df.info())\ndisplay(df.describe(include="all"))\nprint(df.isna().sum())'''),
(2,"Очистка и разведочный анализ",["Исключить невозможные значения","Обосновать заполнение пропусков"],f'''import pandas as pd, matplotlib.pyplot as plt\ndf=pd.read_csv("{P}ras_sensor_log.csv",parse_dates=["timestamp"])\ncols=df.select_dtypes("number").columns\ndf[cols]=df[cols].interpolate(limit_direction="both")\ndf=df.query("0 < dissolved_oxygen_mgL < 15 and 6 < pH < 9")\ndf[cols].hist(figsize=(10,7)); plt.tight_layout()\ndf.to_csv("ras_sensor_log_clean.csv",index=False)'''),
(3,"Признаки и временное разделение",["Создать временные признаки","Не допустить утечки данных"],f'''import pandas as pd\ndf=pd.read_csv("{P}ras_sensor_log.csv",parse_dates=["timestamp"]).sort_values("timestamp")\ndf["hour"]=df.timestamp.dt.hour; df["do_lag1"]=df.dissolved_oxygen_mgL.shift(1)\ndf=df.dropna(); cut=int(len(df)*.8); train,test=df.iloc[:cut],df.iloc[cut:]\nprint(train.timestamp.max(), test.timestamp.min(), train.shape, test.shape)'''),
(4,"Линейная регрессия массы рыбы",["Построить модель","Интерпретировать MAE и R²"],f'''import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_absolute_error,r2_score\ndf=pd.read_csv("{P}fish_growth_dataset.csv"); df=pd.get_dummies(df,columns=["species"],drop_first=True)\nX=df.drop(columns="weight_g");y=df.weight_g\nXtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)\nm=LinearRegression().fit(Xtr,ytr);p=m.predict(Xte)\nprint("MAE",mean_absolute_error(yte,p),"R2",r2_score(yte,p))'''),
(5,"Нелинейная регрессия и переобучение",["Сравнить степени полинома","Использовать кросс-валидацию"],f'''import pandas as pd\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures,StandardScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import KFold,cross_val_score\ndf=pd.read_csv("{P}fish_growth_dataset.csv");X=df[["age_weeks","mean_water_temp_C","daily_feed_ration_g"]];y=df.weight_g\ncv=KFold(5,shuffle=True,random_state=42)\nfor d in [1,2,3]:\n m=make_pipeline(PolynomialFeatures(d),StandardScaler(),Ridge(alpha=1)); s=-cross_val_score(m,X,y,cv=cv,scoring="neg_mean_absolute_error")\n print(d,s.mean(),s.std())'''),
(6,"Регуляризация и итоговая модель",["Настроить гиперпараметры","Сохранить воспроизводимый pipeline"],f'''import pandas as pd, joblib\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\nfrom sklearn.model_selection import GridSearchCV\ndf=pd.read_csv("{P}fish_growth_dataset.csv");X=df[["age_weeks","mean_water_temp_C","daily_feed_ration_g","length_cm"]];y=df.weight_g\ng=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),{{"ridge__alpha":[.1,1,10,100]}},cv=5,scoring="neg_mean_absolute_error").fit(X,y)\nprint(g.best_params_, -g.best_score_); joblib.dump(g.best_estimator_,"growth_model.joblib")'''),
(7,"Классификация производственного риска",["Учесть дисбаланс классов","Оценить редкий класс риска"],f'''import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report\ndf=pd.read_csv("{P}sturgeon_risk_classification.csv");X=df.drop(columns="risk");y=df.risk\nXtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)\nm=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr,ytr)\nprint(classification_report(yte,m.predict(Xte),target_names=["normal","risk"]))'''),
(8,"Порог решения и цена ошибки",["Сравнить пороги","Выбрать порог по recall риска"],f'''import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import precision_recall_fscore_support\ndf=pd.read_csv("{P}sturgeon_risk_classification.csv");X=df.drop(columns="risk");y=df.risk\nXtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)\nm=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xtr,ytr); pr=m.predict_proba(Xte)[:,1]\nfor tau in [.3,.5,.7]:\n p,r,f,_=precision_recall_fscore_support(yte,pr>=tau,average="binary",zero_division=0);print(tau,p,r,f)'''),
(9,"Сравнение классификаторов",["Сравнить модели одинаковым протоколом","Сообщить macro-F1 и PR-AUC"],f'''import pandas as pd\nfrom sklearn.model_selection import StratifiedKFold,cross_validate\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\ndf=pd.read_csv("{P}sturgeon_risk_classification.csv");X=df.drop(columns="risk");y=df.risk\ncv=StratifiedKFold(5,shuffle=True,random_state=42)\nfor name,m in [("LR",LogisticRegression(max_iter=2000,class_weight="balanced")),("RF",RandomForestClassifier(200,class_weight="balanced",random_state=42))]:\n s=cross_validate(m,X,y,cv=cv,scoring=["f1_macro","average_precision"]); print(name,s["test_f1_macro"].mean(),s["test_average_precision"].mean())'''),
(10,"Базовые прогнозы временного ряда",["Создать честный baseline","Использовать хронологический test"],f'''import pandas as pd\nfrom sklearn.metrics import mean_absolute_error\ndf=pd.read_csv("{P}hydrochem_timeseries.csv",parse_dates=["timestamp"]).set_index("timestamp")\ns=df.dissolved_oxygen_mgL; pred=s.shift(1); cut=int(len(s)*.8)\nprint("Test MAE",mean_absolute_error(s.iloc[cut:],pred.iloc[cut:]))'''),
(11,"ARIMA прогноз кислорода",["Прогнозировать 12 часов","Сравнить с baseline"],f'''import pandas as pd\nfrom statsmodels.tsa.arima.model import ARIMA\nfrom sklearn.metrics import mean_absolute_error\ndf=pd.read_csv("{P}hydrochem_timeseries.csv",parse_dates=["timestamp"]).set_index("timestamp").asfreq("h")\ns=df.dissolved_oxygen_mgL; train,test=s.iloc[:-120],s.iloc[-120:]\npreds=[]\nfor i in range(0,120,12):\n hist=s.iloc[:len(train)+i]; preds.extend(ARIMA(hist,order=(2,1,2)).fit().forecast(12))\nprint("ARIMA MAE",mean_absolute_error(test,preds),"Naive MAE",mean_absolute_error(test,s.shift(1).iloc[-120:]))'''),
(12,"LSTM без утечки данных",["Масштабировать только по train","Сравнить на том же тестовом периоде"],f'''import pandas as pd, numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\ndf=pd.read_csv("{P}hydrochem_timeseries.csv",parse_dates=["timestamp"]).set_index("timestamp")\na=df[["temperature_C","dissolved_oxygen_mgL","pH","ammonia_mgL"]].values; cut=int(len(a)*.8)\nsc=MinMaxScaler().fit(a[:cut]); z=sc.transform(a)\nW,H=24,6; X=[];y=[]\nfor i in range(len(z)-W-H+1): X.append(z[i:i+W]);y.append(z[i+W+H-1,1])\nX,y=np.array(X),np.array(y); split=max(0,cut-W-H+1)\nprint("Train/test windows",X[:split].shape,X[split:].shape)\n# Обучение TensorFlow выполняется студентом; сравнение проводится на тех же временных точках, что и ARIMA.'''),
(13,"Предобработка и контуры рыбы",["Исследовать изображения","Создать воспроизводимый CV baseline"],'''from pathlib import Path\nimport cv2, matplotlib.pyplot as plt\np=next(Path("../../data/cv/images/test").glob("*.png")); im=cv2.imread(str(p)); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)\n_,mask=cv2.threshold(g,145,255,cv2.THRESH_BINARY); cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\nprint("Contours",len([c for c in cnt if cv2.contourArea(c)>80])); plt.imshow(mask,cmap="gray");plt.axis("off");'''),
(14,"IoU и mAP на независимом test",["Понять метрики детекции","Не путать IoU одного объекта и mAP"],'''import pandas as pd, numpy as np\na=pd.read_csv("../../data/cv/annotations.csv"); test=a[a.split=="test"].copy()\ndef iou(a,b):\n x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;return inter/u\nr=np.random.default_rng(42); scores=[]\nfor _,q in test.iterrows():\n gt=q[["x1","y1","x2","y2"]].to_numpy(float);pred=gt+r.normal(0,3,4);scores.append(iou(gt,pred))\nprint("Mean IoU",np.mean(scores),"Recall@IoU.50",np.mean(np.array(scores)>=.5))\nprint("Примечание: это учебная симуляция, не результат YOLO и не mAP.")'''),
(15,"YOLOv8 обучение и итоговый pipeline",["Дообучить модель на fry","Оценить только на независимом test"],'''# В Google Colab: раскомментируйте следующую строку\n# !pip install -q ultralytics\nfrom pathlib import Path\nprint("Dataset config:",Path("../../data/cv/fry_dataset.yaml").resolve())\nprint("Train/val/test images:",[len(list(Path(f"../../data/cv/images/{s}").glob("*.png"))) for s in ["train","val","test"]])\n# from ultralytics import YOLO\n# model=YOLO("yolov8n.pt")\n# model.train(data="../../data/cv/fry_dataset.yaml",epochs=30,imgsz=416,seed=42)\n# metrics=model.val(split="test")\n# print(metrics.box.map50, metrics.box.map)\n# Не переносите эти метрики в статью, пока ячейка не выполнена и results.csv не сохранён.''')]

mods={1:"module_1",2:"module_1",3:"module_1",4:"module_2",5:"module_2",6:"module_2",7:"module_3",8:"module_3",9:"module_3",10:"module_4",11:"module_4",12:"module_4",13:"module_5",14:"module_5",15:"module_5"}
for num,title,obj,code in works:
  cells=[("markdown",f"## Цель и результат\n\n{'; '.join(obj)}.\n\n## Самостоятельное задание\n\nИзмените один параметр метода, сравните результат и объясните его значение для аквакультуры.\n\n## Контрольные вопросы\n\n1. Какую ошибку предотвращает использованный протокол?\n2. Как интерпретировать результат для специалиста рыбного хозяйства?\n\n## Оценивание\n\nИспользуйте общую аналитическую рубрику из `assessments/rubric.md`."),("code",code)]
  out=ROOT/f"notebooks/{mods[num]}/PW{num:02}_{title.lower().replace(' ','_')}.ipynb"
  out.write_text(json.dumps(nb(f"Практическая работа {num}. {title}",obj,cells),ensure_ascii=False,indent=1),encoding="utf8")

(ROOT/"requirements.txt").write_text("numpy==2.1.3\npandas==2.2.3\nscikit-learn==1.5.2\nmatplotlib==3.9.2\nstatsmodels==0.14.4\nPillow==11.0.0\nopencv-python-headless==4.10.0.84\njoblib==1.4.2\nultralytics==8.3.0\ntensorflow==2.18.0\n",encoding="utf8")
(ROOT/"data/README.md").write_text("# Data provenance\n\nAll files in this folder are synthetic teaching data generated with seed 2607 by `build_repository.py`. They are not production records and must not be described as real farm data. The CV data contain programmatically drawn fish-like ellipses for instruction only.\n",encoding="utf8")
shutil.copy2("build_repo.py",ROOT/"build_repository.py")

readme='''# Aquaculture AI Course — 15 Practical Works\n\nReproducible supplementary materials for a pilot course in AI and computer vision for aquaculture undergraduates. The repository contains 15 Russian-language Jupyter notebooks, synthetic teaching datasets, assessments, and an article-alignment matrix.\n\n## Scope and evidence statement\n\nThese materials document the intervention. They do not by themselves prove educational effectiveness, transferability, or scalability. All datasets are synthetic. Report student outcomes only from retained individual-level records and report model metrics only from saved, reproducible runs.\n\n## Course structure\n\n| Module | Works | Focus |\n|---|---|---|\n| Data foundations | 1–3 | profiling, cleaning, leakage-safe splitting |\n| Regression | 4–6 | biomass prediction, nonlinearity, regularization |\n| Classification | 7–9 | risk classification, thresholds, model comparison |\n| Time series | 10–12 | baseline, ARIMA, LSTM preparation |\n| Computer vision | 13–15 | OpenCV, detection metrics, YOLOv8 |\n\n## Use\n\nInstall `requirements.txt`, open the repository root, and run notebooks in numeric order. Paths assume execution from each notebook's module folder. In Colab, upload the repository or mount Drive without changing its structure. YOLOv8 training requires internet access to obtain pretrained weights and benefits from a GPU.\n\n## Reproducibility\n\nThe tabular and image datasets are included. The final YOLO notebook deliberately does not report a predetermined mAP. Save `results.csv`, model weights, environment details, and the exact test metrics after a real run before citing them.\n\n## License\n\nCode and original teaching text: MIT. Synthetic datasets: CC0.\n'''
(ROOT/"README.md").write_text(readme,encoding="utf8")

rub='''# Analytical rubric\n\nEach practical work is scored from 0 to 16.\n\n| Criterion | 0 | 1 | 2 | 3 | 4 |\n|---|---|---|---|---|---|\n| Code correctness | absent | does not run | runs with major help | runs with minor fixes | fully reproducible |\n| Method choice | absent | incorrect | partly justified | correct | compared with alternatives |\n| Interpretation | absent | incorrect | generic | domain-specific | includes uncertainty and risk |\n| Documentation | absent | incomplete | basic | clear | enables independent reproduction |\n\nConvert to 100 points as `score / 16 × 100`. Two assessors independently score a calibration sample; report inter-rater agreement before outcome analysis.\n'''
(ROOT/"assessments/rubric.md").write_text(rub,encoding="utf8")
items=[]
constructs=["Data literacy","Regression","Classification","Time series","Computer vision"]
for i in range(20): items.append([i+1,constructs[i%5],f"Item {i+1}: applied concept question",0,4])
pd.DataFrame(items,columns=["item","construct","prompt_placeholder","min_score","max_score"]).to_csv(ROOT/"assessments/pre_post_blueprint.csv",index=False)
(ROOT/"assessments/README.md").write_text("# Assessment package\n\nUse parallel pre/post forms mapped to `pre_post_blueprint.csv`; do not use Work 1 and Work 15 as pre/post tests. Retain item-level anonymized responses, assessor IDs, missingness, and timestamps. Validate content with domain and education experts before deployment.\n",encoding="utf8")

rows=[]
for n,title,_,_ in works: rows.append([n,mods[n].replace("module_","Module "),title,"Notebook and synthetic data included","Do not claim causal effectiveness without a comparison group"])
pd.DataFrame(rows,columns=["work","module","title","evidence","claim_limit"]).to_csv(ROOT/"docs/article_alignment.csv",index=False)
(ROOT/"docs/ARTICLE_REVISION_NOTES.md").write_text('''# Required manuscript revisions\n\n1. Replace every reference to 30 works with 15 and state three works per module.\n2. State that the teaching datasets are synthetic unless separately documented production data were actually used.\n3. Change the mid-test point to after Work 9.\n4. Do not report mAP 0.92 unless a saved independent-test run supports it.\n5. Provide n, mean, SD, confidence intervals, inferential tests, effect sizes, missing-data handling, ethics, and instrument evidence.\n6. Without a comparison group, frame the study as a pilot single-group pre/post study and avoid causal, transferability, and scalability claims.\n''',encoding="utf8")
(ROOT/"LICENSE").write_text("MIT License\n\nCopyright (c) 2026 Course authors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, subject to the conditions of the MIT License.\n",encoding="utf8")

zip_path=Path("aquaculture-ai-15-ready.zip")
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
  for p in ROOT.rglob("*"):
    if p.is_file(): z.write(p,p.as_posix())
print(ROOT,zip_path)
