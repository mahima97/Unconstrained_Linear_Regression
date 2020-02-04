from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
import sys
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from flask import Flask, request, jsonify


def DataPrep(df):
	drop_thresh = df.shape[0]*.9
	df = df.dropna(thresh=drop_thresh, how='all', axis='columns').copy()

	# fill the values using forward fill method
	methods = ['backfill', 'bfill', 'pad', 'ffill', None]



	df = df.fillna(method = methods[3])
	names = df.select_dtypes(include=['object']).columns
	# print(names)

	# cat_thresh = df.shape[0]*.7
	cat_thresh = 32
	for col in df.columns:
		if df[col].nunique() < cat_thresh and col in names:
			print(col,df[col].nunique())
			df[col] = df[col].astype('category')


	cat_names = df.select_dtypes(include=['category']).columns
	categorical_cols = df.select_dtypes(include=['category'])
	# print(categorical_cols)

	df_enc = pd.DataFrame()
	le = LabelEncoder()
	for col in categorical_cols.columns:
		df_enc[col] = le.fit_transform(categorical_cols[col])

	# print(df_enc)

	df1 = df.select_dtypes(include=['float64','int']).copy()

	# print(df1.shape,df_enc.shape)
	df_all = pd.concat([df1,df_enc],axis=1)
	return df_all


app = Flask(__name__)

@app.route("/LinearReg",methods=['POST'])

def LinearReg():
	# try:
		# try:

	jason_ = request.json
	# print("JASON REQUEST:",jason_)
	fname   = request.json["fname"]
	target = request.json["target"]
	# print(fname[0],target_lst[0])
	df = pd.read_csv(fname)

	##to quickly drop all the columns where at least 90% of the data is empty

	# df_all = df_all.dropna()
	# print(df_all)
	df_all = DataPrep(df)
	# target = target_lst[0]



	y = df_all[target]
	X = df_all.drop(columns = target,axis=1)
	attributes = X.columns.tolist()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


	# pipeline = make_pipeline(Normalizer(), LinearRegression())
	pipeline = Pipeline([('scl', Normalizer()),
                    ('clf', LinearRegression(n_jobs=-1))])

	reg = pipeline.fit(X_train, y_train)

	result = reg.predict(X)
	score = reg.score(X_test,y_test)
	# print(score)
	coef = reg.named_steps['clf'].coef_
	coef = coef.tolist()
	intercept = reg.named_steps['clf'].intercept_
	# print(coef,intercept)

	# print(reg..intercept_)
	# print()
	df['output'] = result
	json_out = df.to_json(orient='records')
	# print(df['output'])
	df.to_csv(fname[:-4]+'_output.csv',index=False)
	return jsonify({'score':score,
		'equation_coef':coef,
		'attributes':attributes,
		'intercept':intercept,
		'output_name':fname[:-4]+'_output.csv'})

	# except:
	# 	print("*"*50 +"ERROR"+ "*"*50)

	# 	return jsonify({"ERROR": "Something is wrong"})
	# fname = sys.argv[1]
	# target = sys.argv[2]

	# fname = 'gld_price_data.csv'
	# target = 'GLD'

	# print(target)


if __name__ == '__main__':

	app.run(port=1234, debug=True)
