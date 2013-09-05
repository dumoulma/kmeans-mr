package ca.ulaval.ift.graal.fic.utils;

import java.io.IOException;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.ValueServer;
import org.apache.commons.math3.random.Well19937c;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * This class is used to create the date to run the algorithm over. The
 * centroids are all hard coded. Next step would be to use the canopy method to
 * determine the number of clusters automatically.
 * 
 * @author dumoulma
 * 
 */
public class DataGenerator {
    public static void main(String[] args) throws IOException {
        Path centroidOutPath = new Path("data/kmeans/centroid.seq");
        Path dataOutPath = new Path("data/kmeans/data.seq");
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(dataOutPath))
            fs.delete(centroidOutPath, true);
        if (fs.exists(centroidOutPath))
            fs.delete(dataOutPath, true);

        SequenceFile.Writer centroidOut = null;
        SequenceFile.Writer dataOut = null;

        RandomGenerator randomData = new Well19937c();
        ValueServer valueServer = new ValueServer(randomData);
        valueServer.setSigma(2.5);
        try {
            centroidOut = SequenceFile.createWriter(fs, conf, centroidOutPath, IntWritable.class,
                    VectorWritable.class);
            dataOut = SequenceFile.createWriter(fs, conf, dataOutPath, LongWritable.class,
                    VectorWritable.class);

            double[][] centroids = { { 1.0, 1.0 }, { 1.0, 5.0 }, { 2.5, 1.0 }, { 2.5, 5.0 },
                    { 5.0, 1.0 }, { 5.0, 5.0 } };
            for (int i = 0; i < centroids.length; i++) {
                for (int j = 0; j < 100000; j++) {
                    double[] values = new double[2];
                    values[0] = randomData.nextGaussian() + centroids[i][0];
                    values[1] = randomData.nextGaussian() + centroids[i][1];
                    Vector v = new DenseVector(values);
                    dataOut.append(new LongWritable((i + 1) * j), new VectorWritable(v));
                }
                centroidOut.append(new IntWritable(i), new VectorWritable(new DenseVector(
                        centroids[i])));
            }

        } finally {
            if (centroidOut != null) {
                centroidOut.close();
            }
            if (dataOut != null) {
                dataOut.close();
            }
        }
    }
}
