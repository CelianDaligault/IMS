from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg, stddev
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("WeatherPrediction").getOrCreate()
df = spark.read.csv("weather_data.csv", header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp(col("utc_timestamp"))).drop("utc_timestamp")
df = df.dropna()

df_grouped = df.groupBy("timestamp").agg(
    avg("FR_temperature").alias("mean_temp"),
    stddev("FR_temperature").alias("stddev_temp"),
    avg("FR_radiation_direct_horizontal").alias("mean_radiation_direct"),
    avg("FR_radiation_diffuse_horizontal").alias("mean_radiation_diffuse")
)


assembler = VectorAssembler(inputCols=["mean_temp", "mean_radiation_direct", "mean_radiation_diffuse"], outputCol="features")
df_ml = assembler.transform(df_grouped)
train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

model = RandomForestRegressor(featuresCol="features", labelCol="mean_temp", numTrees=10)
trained_model = model.fit(train)
predictions = trained_model.transform(test)
df_pandas = df_grouped.toPandas()
predictions_pd = predictions.select("timestamp", "mean_temp", "prediction").toPandas()

plt.figure(figsize=(10, 5))
plt.plot(predictions_pd["timestamp"], predictions_pd["mean_temp"], label="Real Mean Temperature", color='b')
plt.plot(predictions_pd["timestamp"], predictions_pd["prediction"], label="Predicted Mean Temperature", color='r', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.grid()
plt.savefig("temperature_predictions_plot.png")
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_pandas["timestamp"], df_pandas["mean_temp"], label="Mean Temperature FR", color='b')
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.legend()
plt.grid()
plt.savefig("temperature_plot.png")
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_pandas["timestamp"], df_pandas["mean_radiation_direct"], label="Mean Direct Radiation", color='r')
plt.xlabel("Date")
plt.ylabel("Mean Direct Radiation")
plt.legend()
plt.grid()
plt.savefig("radiation_direct_plot.png")
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_pandas["timestamp"], df_pandas["mean_radiation_diffuse"], label="Mean Diffuse Radiation", color='g')
plt.xlabel("Date")
plt.ylabel("Mean Diffuse Radiation")
plt.legend()
plt.grid()
plt.savefig("radiation_diffuse_plot.png")
# plt.show()

with open("README.md", "w") as f:
    f.write("# Weather Prediction Report\n\n")
    f.write("## Summary\n")
    f.write("This project uses PySpark to analyze and predict temperature trends based on weather data.\n\n")
    f.write("## Results\n")
    f.write("### Mean Temperature Trends\n")
    f.write("![Temperature Trends](temperature_plot.png)\n\n")
    f.write("### Mean Direct Radiation Trends\n")
    f.write("![Direct Radiation Trends](radiation_direct_plot.png)\n\n")
    f.write("### Mean Diffuse Radiation Trends\n")
    f.write("![Diffuse Radiation Trends](radiation_diffuse_plot.png)\n\n")
    f.write("### Model Performance\n")
    predictions_str = predictions.select("mean_temp", "prediction").toPandas().to_string(index=False)
    f.write(predictions_str)