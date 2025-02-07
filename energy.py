from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("WeatherPrediction").getOrCreate()

df = spark.read.csv("when2heat_filtered.csv", header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp(col("utc_timestamp"))).drop("utc_timestamp")
df = df.dropna() 


df_grouped = df.groupBy("timestamp").agg(
    avg("FR_heat_demand_water").alias("energy_demand_water"),
    avg("FR_heat_demand_space").alias("energy_demand_space"),
    avg("FR_heat_demand_total").alias("energy_demand_total"),
)

df_pandas = df_grouped.toPandas()

assembler = VectorAssembler(inputCols=["energy_demand_water", "energy_demand_space", "energy_demand_total"], outputCol="features")
df_ml = assembler.transform(df_grouped)

train, test = df_ml.randomSplit([0.9, 0.1], seed=42)

model = RandomForestRegressor(featuresCol="features", labelCol="energy_demand_total", numTrees=10)
trained_model = model.fit(train)

predictions = trained_model.transform(test)

predictions_pd = predictions.select("timestamp", "energy_demand_total", "prediction").toPandas()

predictions_pd['error'] = abs(predictions_pd['energy_demand_total'] - predictions_pd['prediction'])

threshold = 10000

alerts = predictions_pd[predictions_pd['error'] > threshold]

if not alerts.empty:
    print(f"Alert: There are {len(alerts)} instances where the prediction error exceeds {threshold}°C.")
    print("Details of the alerts:")
    print(alerts[['timestamp', 'energy_demand_total', 'prediction', 'error']])

plt.figure(figsize=(10, 5))
plt.scatter(predictions_pd["timestamp"], predictions_pd["energy_demand_total"], label="Real energy demand", color='b', s=5)
plt.scatter(predictions_pd["timestamp"], predictions_pd["prediction"], label="Predicted energy", color='r', s=5)
plt.xlabel("Date")
plt.ylabel("Energy demand Real vs Predicted (in MW)")
plt.title("Temperature: Real vs Predicted")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Prediction_vs_reality.png")
plt.show()

plt.scatter(df_pandas["timestamp"], df_pandas["energy_demand_water"], label="energy_demand water", color='b', s=5)
plt.xlabel("Date")
plt.ylabel("Energy_demand consumption (in MW)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("energy_demand_water.png")

plt.scatter(df_pandas["timestamp"], df_pandas["energy_demand_space"], label="energy_demand space", color='r', s=5)
plt.xlabel("Date")
plt.ylabel("Energy_demand consumption (in MW)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("energy_demand_space.png")


plt.scatter(df_pandas["timestamp"], df_pandas["energy_demand_total"], label="energy_demand total", color='g', s=5)
plt.xlabel("Date")
plt.ylabel("Energy_demand consumption (in MW)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("energy_demand_space.png")

with open("README.md", "w") as f:
    f.write("# Energy demand Report\n\n")
    f.write("## Summary\n")
    f.write("This project uses PySpark to analyze and predict Energy demand in France.\n\n")
    f.write("## Results\n")
    f.write("### Energy water\n")
    f.write("![Energy water](temperature_plot.png)\n\n")
    f.write("### Energy space\n")
    f.write("![Energy space](radiation_direct_plot.png)\n\n")
    f.write("### Energy total\n")
    f.write("![Energy total](radiation_diffuse_plot.png)\n\n")
    f.write("### Model Performance\n")