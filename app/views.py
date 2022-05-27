from app import app
from flask import render_template, request
import pickle
from sklearn.neighbors import NearestNeighbors


@app.route("/")
def index():
    with open(r'app\static\pickle\df1.pkl', "rb") as input_file:
        dataFrame_1=pickle.load(input_file)
    car_name=dataFrame_1['car_name'].values.tolist()
    return render_template('index.html',car_name=car_name)

@app.route("/content-based-search",methods = ['POST'])
def content_based_search():
    if request.method == 'POST':
      result = request.form
    #   print(result['keyword'])
      return recommend(result['keyword'])

@app.route("/item-based-search",methods = ['POST'])
def item_based_search():
    if request.method == 'POST':
      result = request.form
    #   print(result['keyword'])
      return recommend_knn(result['highway_mileage'],result['city_mileage'],result['price'],result['companyRatingvalue'],result['consumerRatingvalue'])

def recommend_knn(highway_mileage,city_mileage,price,companyRatingvalue,consumerRatingvalue):
    with open(r'app\static\pickle\csr_sample.pkl', "rb") as input_file:
        csr_sample=pickle.load(input_file)
    input_file.close()
    with open(r'app\static\pickle\df1.pkl', "rb") as input_file:
        cars_data=pickle.load(input_file)
    input_file.close()
    cars_data.rename(columns={'name':'cname'},inplace=True)
    with open(r'app\static\pickle\df2.pkl', "rb") as input_file:
        df=pickle.load(input_file)
    input_file.close()

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    knn.fit(csr_sample)
    dataset_sort_des = df.sort_values(['rating','city_milege','highway_milege','msrp','consumer_rating'], ascending=[True,True,True,True,True])
    # filter1 = dataset_sort_des.loc[((dataset_sort_des['rating'] == companyRatingvalue)  | (dataset_sort_des['consumer_rating'] == consumerRatingvalue) )  | ((dataset_sort_des['city_milege'] == city_mileage) | (dataset_sort_des['highway_milege'] == highway_mileage))  | (dataset_sort_des['msrp'] == price) ].row_num
    filter1 = dataset_sort_des.loc[ (dataset_sort_des['consumer_rating'] == float(consumerRatingvalue))  | (dataset_sort_des['rating'] == float(companyRatingvalue) ) | ( ( dataset_sort_des['msrp'] == float(price)) ) |( (dataset_sort_des['city_milege'] == float(city_mileage)) | (dataset_sort_des['highway_milege'] == float(highway_mileage))) ].row_num
    filter1 = filter1.tolist()

    #print(len(filter1) )
 

    distances1=[]
    indices1=[]
  
    for i in filter1:
       
        distances , indices = knn.kneighbors(csr_sample[i],n_neighbors=5)
        indices = indices.flatten()
        indices= indices[1:]
        indices1.extend(indices)
    
       

    df1=df[df['row_num'].isin(indices1)]
    recom_list=df1['ID'].values.tolist()[1:11]
    result={}
    if len(recom_list)>0:
        list_of_cars=[]
        for val in recom_list:
            df2=cars_data.loc[cars_data['ID']==val]
            # print(df2)
            temp={}
            temp['car name']=df2.cname.values[0]
            temp['brand name']=df2.brand_name.values[0]
            temp['engine']=df2.engine_cc.values[0]
            temp['submodel']=df2.submodel.values[0]
            temp['msrp']=df2.msrp.values[0]
            temp['photo']=df2.photo.values[0]
            temp['hm']=df2.highway_milege.values[0]
            temp['cm']=df2.city_milege.values[0]
            temp['conrat']=df2.consumer_rating.values[0]
            temp['comrat']=df2.rating.values[0]

            list_of_cars.append(temp)
        result['data']=list_of_cars
        result['Status']=200
        result['msg']="successfull"
    else:
        result['data']=[]
        result['Status']=404
        result['msg']="Cars not Found!!!"
    return result


def recommend(car):
    with open(r'app\static\pickle\df1.pkl', "rb") as input_file:
        cars_data=pickle.load(input_file)
    input_file.close()

    cars_data.rename(columns={'name':'cname'},inplace=True)
    # print(cars_data.columns)

    with open(r'app\static\pickle\similarity.pkl', "rb") as input_file:
        similarity=pickle.load(input_file)
    input_file.close()
    car_record=cars_data[cars_data['car_name'] == car]
    result={}
    if len(car_record)>0:
        car_index = car_record.index[0]
        distances = similarity[car_index]
        cars_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:7]
        
        list_of_cars=[]
        for i in cars_list:
            # print(cars_data.iloc[i[0]])
            temp={}
            temp['car name']=cars_data.iloc[i[0]].cname
            temp['brand name']=cars_data.iloc[i[0]].brand_name
            temp['engine']=cars_data.iloc[i[0]].engine_cc
            temp['submodel']=cars_data.iloc[i[0]].submodel
            temp['msrp']=cars_data.iloc[i[0]].msrp
            temp['photo']=cars_data.iloc[i[0]].photo

            list_of_cars.append(temp)
        result['data']=list_of_cars
        result['Status']=200
        result['msg']="successfull"
    else:
        result['data']=[]
        result['Status']=404
        result['msg']="Cars not Found!!!"
    return result
