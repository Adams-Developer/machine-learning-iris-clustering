using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace irisFloweringClustering
{
    /* input data class with definitions for
     * each feature from the data set.
     * Column attribute specifies the indices
     * of the source columns.
     * 
     * the float type represents floating-point
     * values in the input and prediction
     * data classes.
     */
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

    }

    /*
     * This class represents the output of 
     * the clustering model appled to an IrisData
     * instance.
     * 
     * PredictedLabel column contains the Id of the
     * predicted cluster.
     * The Score column contains an array with squared
     * Euclidean distances to the cluster centroids. 
     * 
     * The array length is equal to the number of 
     * clusters
     */
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }

}
