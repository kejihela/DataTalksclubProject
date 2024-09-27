import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

reference_data = pd.read_cv("data/reference.csv")

with open("models/linear_reg.bin",'rb') as f_in:
	model = joblib.load(f_in)
	
raw_data = pd.read_csv("https://raw.githubusercontent.com/kejihela/DataTalksclubProject/master/dataset/flight_dataset.csv")
categorical = ["Airline", "Source", "Destination"]
numerical = ["Total_Stops","Duration_hours","Duration_min"]

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical,
    categorical_features=categorical
)
report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
]
)

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""

def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

def calculate_dummy_metrics_postgresql(curr):
	raw_data = raw_data[categorical + numerical]
	enc = OrdinalEncoder()
    enc.fit(raw_data[["Airline","Source", "Destination"]])
    raw_data[["Airline","Source", "Destination"]] = enc.transform(raw_data[["Airline","Source", "Destination"]])
	current_data = [raw_data]
    current_data['prediction'] = model.predict(current_data )


    report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
 
	curr.execute(
		"insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), prediction_drift, num_drifted_columns, share_missing_values)
	)

def main():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, 100):
			with conn.cursor() as curr:
				calculate_dummy_metrics_postgresql(curr)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	main()