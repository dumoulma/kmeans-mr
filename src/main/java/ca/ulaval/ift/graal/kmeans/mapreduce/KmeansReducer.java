package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KmeansReducer extends
        Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private static final Logger LOG = LoggerFactory.getLogger(KmeansReducer.class);
    private static double convergence_threshold = 0.01;

    public static enum Counter {
        NOT_CONVERGED, CONVERGED
    }

    private Map<Integer, Vector> centers = new HashMap<Integer, Vector>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        Configuration conf = context.getConfiguration();
        centers = MRUtils.readClusters(conf);
        convergence_threshold = conf.getFloat("error.threshold", (float) convergence_threshold);

        LOG.debug("Convergence threshold set to: " + convergence_threshold);
        LOG.debug("Reducer Centers read: " + centers.size());
    }

    @Override
    protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context)
            throws IOException, InterruptedException {
        Vector currentCenter = centers.get(key.get());
        Vector newCenter = computeNewCenter(key, values);

        LOG.info("Current center value:" + currentCenter + "\n" + "new center value: " + newCenter);

        if (isConverged(currentCenter, newCenter)) {
            context.getCounter(Counter.CONVERGED).increment(1);
            LOG.info("CONVERGED! Center Index: " + key.get());
        }
        context.write(key, new VectorWritable(newCenter));
    }

    private Vector computeNewCenter(IntWritable key, Iterable<VectorWritable> values) {
        Vector newCenter = null;
        long valuesCount = 0;
        for (VectorWritable value : values) {
            DenseVector nextVector = (DenseVector) value.get();
            if (newCenter == null)
                newCenter = nextVector.like();
            newCenter = newCenter.plus(nextVector);
            valuesCount++;
        }

        return newCenter.divide(valuesCount);
    }

    private boolean isConverged(Vector centroid, Vector newCentroid) {
        double lengthSquared = Math.sqrt(centroid.minus(newCentroid).getLengthSquared());
        return lengthSquared < convergence_threshold;
    }

    public KmeansReducer() {
        super();
    }

    // NOTE: for testing purposes ONLY!
    KmeansReducer(Map<Integer, Vector> centers) {
        this.centers = centers;
    }

}
