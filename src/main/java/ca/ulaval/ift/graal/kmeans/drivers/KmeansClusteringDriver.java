package ca.ulaval.ift.graal.kmeans.drivers;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobStatus;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ca.ulaval.ift.graal.fic.utils.Util;
import ca.ulaval.ift.graal.kmeans.mapreduce.KmeansMapper;
import ca.ulaval.ift.graal.kmeans.mapreduce.KmeansReducer;

/**
 * K-Means clustering demo code
 * 
 * This driver iterates over the data to improve the cluster centroids until an
 * iteration cutoff or converge
 * 
 * <p>
 * Based in part on: <a href=
 * "http://codingwiththomas.blogspot.ca/2011/05/k-means-clustering-with-mapreduce.html"
 * >Thomas Jungblut's blog post</a>
 * <p>
 */
public class KmeansClusteringDriver extends Configured implements Tool {
    private static final Logger LOG = LoggerFactory.getLogger(KmeansClusteringDriver.class);

    private static final String DATA_PATH = "data/kmeans/data.seq";
    private static final String INITIAL_CLUSTERS_PATH = "data/kmeans/centroid.seq";
    private static final int MAX_ITERATIONS = 20;
    private static final int K = 6;

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new KmeansClusteringDriver(), args);
        System.exit(exitCode);
    }

    @Override
    public int run(String[] arg0) throws IOException, ClassNotFoundException, InterruptedException {
        LOG.info("Demo for Kmeans clustering Job -- run()");

        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        conf.setFloat("error.threshold", 0.005f);
        Path dataPath = new Path(DATA_PATH);

        long oldCounter = 0;
        int iteration = 1;
        boolean iterationWasSuccessful = true;
        boolean hasConverged = false;
        Path centroidPath = new Path(INITIAL_CLUSTERS_PATH);
        while (!hasConverged && iteration < MAX_ITERATIONS && iterationWasSuccessful) {
            LOG.info("Kmeans iteration #" + iteration);

            conf.set("centroid.path", centroidPath.toString());
            conf.setInt("current.iteration", iteration);

            Job job = new Job(conf);
            job.setJobName("KMeans Clustering depth_" + iteration);
            job.setJarByClass(getClass());

            job.setMapperClass(KmeansMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(VectorWritable.class);
            job.setInputFormatClass(SequenceFileInputFormat.class);
            FileInputFormat.addInputPath(job, dataPath);

            job.setReducerClass(KmeansReducer.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(VectorWritable.class);
            job.setOutputFormatClass(SequenceFileOutputFormat.class);
            Path nextCentroidPath = new Path("data/kmeans/depth_" + iteration);
            if (fs.exists(nextCentroidPath))
                fs.delete(nextCentroidPath, true);
            FileOutputFormat.setOutputPath(job, nextCentroidPath);

            iterationWasSuccessful = job.waitForCompletion(true);

            iteration++;
            centroidPath = nextCentroidPath;
            nextCentroidPath = new Path("data/kmeans/depth_" + iteration);
            if (fs.exists(nextCentroidPath))
                fs.delete(nextCentroidPath, true);
            long convergenceCounter = job.getCounters()
                    .findCounter(KmeansReducer.Counter.CONVERGED).getValue();
            LOG.info("KmeansReducer Counter: " + convergenceCounter);

            if (convergenceCounter - oldCounter == K) {
                hasConverged = true;
                LOG.info("Converged!!");
            }
        }

        Util.showClusters(conf, centroidPath);

        return JobStatus.SUCCEEDED;
    }
}
