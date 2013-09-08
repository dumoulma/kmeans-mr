package ca.ulaval.ift.graal.kmeans.drivers;

import java.io.IOException;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.ValueServer;
import org.apache.commons.math3.random.Well19937c;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobStatus;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is used to create the date to run the algorithm over. The
 * centroids are all hard coded. Next step would be to use the canopy method to
 * determine the number of clusters automatically.
 * 
 * @author Mathieu Dumoulin
 * 
 */
public class DataGenerator extends Configured implements Tool {

    private static final Logger LOG = LoggerFactory.getLogger(DataGenerator.class);

    // change this to generate different centroids. This could be
    // improved by generating random centroid vectors of a specified length or
    // reading them from a file
    static final double[][] centroids = { { 1.0, 1.0, 1.0, 1.0 }, { 1.0, 5.0, 1.0, 2.0 },
            { 2.5, 1.0, 2.5, 1.0 }, { 2.5, 5.0, 1.0, 1.5 }, { 5.0, 1.0, 5.0, 2.5 },
            { 5.0, 5.0, 5.0, 5.0 } };

    // This value should match the length of the vectors defined in centroids!
    private static final int VECTOR_LENGTH = 4;

    // a smaller value of sigma will make the point clouds smaller, the
    // centroids will be easier to find
    private static final double SIGMA = 3;

    // for each centroid, how many points to generate (should be very large if
    // run on a cluster)
    private static final int N_POINTS = 100000;

    private static final String DATA_PATH = "data/kmeans/data.seq";
    private static final String INITIAL_CLUSTERS_PATH = "data/kmeans/centroid.seq";

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new DataGenerator(), args);
        System.exit(exitCode);
    }

    @Override
    public int run(String[] arg0) throws IOException, ClassNotFoundException, InterruptedException {
        LOG.info("JOB START");

        Path centroidOutPath = new Path(INITIAL_CLUSTERS_PATH);
        Path dataOutPath = new Path(DATA_PATH);
        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(dataOutPath))
            fs.delete(centroidOutPath, true);
        if (fs.exists(centroidOutPath))
            fs.delete(dataOutPath, true);

        SequenceFile.Writer centroidOut = null;
        SequenceFile.Writer dataOut = null;

        RandomGenerator randomData = new Well19937c();
        ValueServer valueServer = new ValueServer(randomData);
        valueServer.setSigma(SIGMA);
        try {
            centroidOut = SequenceFile.createWriter(fs, conf, centroidOutPath, IntWritable.class,
                    VectorWritable.class);
            dataOut = SequenceFile.createWriter(fs, conf, dataOutPath, LongWritable.class,
                    VectorWritable.class);

            for (int i = 0; i < centroids.length; i++) {
                for (int j = 0; j < N_POINTS; j++) {
                    double[] values = new double[VECTOR_LENGTH];
                    for (int k = 0; k < VECTOR_LENGTH; k++) {
                        values[k] = randomData.nextGaussian() + centroids[i][k];
                    }

                    Vector v = new DenseVector(values);
                    dataOut.append(new LongWritable((i + 1) * j), new VectorWritable(v));

                    LOG.debug("New vector added: " + v.toString());
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

        LOG.info("SUCCEEDED!");

        return JobStatus.SUCCEEDED;
    }
}
