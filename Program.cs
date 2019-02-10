using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;


namespace irisFloweringClustering
{
    class Program
    {
        //Defining data and model paths
        /* Fields to hold the paths to the data set file
         * and to the file to save the model.
         * 
         * _dataPath contains the path to the file with the data set used to train the model.
         * _modelPath contains the path to the file where the trained model is stored.
         */

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris-data.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        static void Main(string[] args)
        {
            //Creates ML Context enviornment and provides mechanisms for logging
            //and entry points for data loading, model training, prediction, and other tasks.
            var mlContext = new MLContext(seed: 0);

            //Setup the way to load data
            TextLoader textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                //Separator = ",",
                HasHeader = false,
                Column = new[]
                            {
                    new TextLoader.Column("SepalLength", DataKind.R4, 0),
                    new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                    new TextLoader.Column("PetalLength", DataKind.R4, 2),
                    new TextLoader.Column("PetalWidth", DataKind.R4, 3)
                }
            });

            //Instantiate TextLoader instance to create an IDataView instance
            //This represents the data source for the training data set
            IDataView dataView = textLoader.Read(_dataPath);

            
            //Create a learning pipeline
            /* Concatenate loaded columns into one Features column, which is used by a clustering trainer
             * Using a KMPPT trainer to train the model using the k-means++ clustering algorithm
             * 
             * The code specifies that the data set should be split in three clusters
             */          
            string featuresColumnName = "Features";

            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, clustersCount: 3));

            //Train the model
            var model = pipeline.Fit(dataView);

            //Save the model
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fileStream);
            }

            //use the model for predictions
            var predictor = model.CreatePredictionEngine<IrisData, ClusterPrediction>(mlContext);

            //Find out the cluster to which the specified item (from TestIrisData) belongs to
            var prediction = predictor.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");

        }
    }
}
