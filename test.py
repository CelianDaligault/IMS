from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg, stddev
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("WeatherPrediction").getOrCreate()

df = spark.read.csv("weather_data.csv", header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp(col("utc_timestamp"))).drop("utc_timestamp")
df = df.dropna() 

country = 'BG'

df_grouped = df.groupBy("timestamp").agg(
    avg(country+"_temperature").alias("mean_temp"),
    stddev(country+"_temperature").alias("stddev_temp"),
    avg(country+"_radiation_direct_horizontal").alias("mean_radiation_direct"),
    avg(country+"_radiation_diffuse_horizontal").alias("mean_radiation_diffuse")
)

df_pandas = df_grouped.toPandas()

assembler = VectorAssembler(inputCols=["mean_temp", "mean_radiation_direct", "mean_radiation_diffuse"], outputCol="features")
df_ml = assembler.transform(df_grouped)

train, test = df_ml.randomSplit([0.9, 0.1], seed=42)

model = RandomForestRegressor(featuresCol="features", labelCol="mean_temp", numTrees=10)
trained_model = model.fit(train)

predictions = trained_model.transform(test)

predictions_pd = predictions.select("timestamp", "mean_temp", "prediction").toPandas()

predictions_pd['error'] = abs(predictions_pd['mean_temp'] - predictions_pd['prediction'])

threshold = 14

alerts = predictions_pd[predictions_pd['error'] > threshold]

if not alerts.empty:
    print(f"Alert: There are {len(alerts)} instances where the prediction error exceeds {threshold}°C.")
    print("Details of the alerts:")
    print(alerts[['timestamp', 'mean_temp', 'prediction', 'error']])

plt.figure(figsize=(10, 5))
plt.scatter(predictions_pd["timestamp"], predictions_pd["mean_temp"], label="Real Mean Temperature", color='b', s=5)
plt.scatter(predictions_pd["timestamp"], predictions_pd["prediction"], label="Predicted Mean Temperature", color='r', s=5)
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.title("Temperature Trend: Real vs Predicted")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temperature_trend_plot.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df_pandas["timestamp"], df_pandas["mean_radiation_direct"], label="Mean Direct Radiation", color='r', s=5)
plt.xlabel("Date")
plt.ylabel("Mean Direct Radiation")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("radiation_direct_trend_plot.png")

plt.figure(figsize=(10, 5))
plt.scatter(df_pandas["timestamp"], df_pandas["mean_radiation_diffuse"], label="Mean Diffuse Radiation", color='g', s=5)
plt.xlabel("Date")
plt.ylabel("Mean Diffuse Radiation")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("radiation_diffuse_trend_plot.png")

with open("README.md", "w") as f:
    f.write("# Weather Prediction Report\n\n")
    f.write("## Summary\n")
    f.write("This project uses PySpark to analyze and predict temperature trends based on weather data.\n\n")
    f.write("## Results\n")
    f.write("### Temperature Trend\n")
    f.write("![Temperature Trend](temperature_trend_plot.png)\n\n")
    f.write("### Mean Direct Radiation Trend\n")
    f.write("![Direct Radiation Trend](radiation_direct_trend_plot.png)\n\n")
    f.write("### Mean Diffuse Radiation Trend\n")
    f.write("![Diffuse Radiation Trend](radiation_diffuse_trend_plot.png)\n\n")
    f.write("### Model Performance\n")