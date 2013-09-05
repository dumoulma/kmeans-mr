package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KmeansMapper extends Mapper<LongWritable, VectorWritable, IntWritable, VectorWritable> {
    private static final Logger LOG = LoggerFactory.getLogger(KmeansMapper.class);

    private Map<Integer, Vector> centers = new HashMap<Integer, Vector>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        centers = MRUtils.readClusters(context.getConfiguration());

        LOG.info("Mapper Centers read: " + centers.size());
    }

    @Override
    protected void map(LongWritable key, VectorWritable value, Context context) throws IOException,
            InterruptedException {
        Vector point = value.get();
        int nearestCenterIndex = -1;
        double nearestDistance = Double.MAX_VALUE;
        for (Integer index : centers.keySet()) {
            double dist = point.getDistanceSquared(centers.get(index));
            if (dist < nearestDistance) {
                nearestDistance = dist;
                nearestCenterIndex = index;
            }
        }
        context.write(new IntWritable(nearestCenterIndex), value);
    }

    public KmeansMapper() {
        super();
    }

    // NOTE: for testing purposes ONLY!
    KmeansMapper(Map<Integer, Vector> centers) {
        this.centers = centers;
    }
}
