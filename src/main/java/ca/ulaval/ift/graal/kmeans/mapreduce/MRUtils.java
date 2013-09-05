package ca.ulaval.ift.graal.kmeans.mapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class MRUtils {
    private static final Logger LOG = LoggerFactory.getLogger(MRUtils.class);

    static Map<Integer, Vector> readClusters(Configuration conf) throws IOException {
        Map<Integer, Vector> centers = new HashMap<Integer, Vector>();
        Path centroidsPath = new Path(conf.get("centroid.path"));
        FileSystem fs = FileSystem.get(conf);
        List<Path> filePaths = listOutputFiles(fs, centroidsPath);
        if (filePaths.isEmpty())
            LOG.warn("No files found in dir: " + centroidsPath);

        for (Path path : filePaths) {
            LOG.info("FOUND " + path.toString());
            try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf)) {

                IntWritable key = new IntWritable();
                VectorWritable value = new VectorWritable();
                while (reader.next(key, value)) {
                    int index = key.get();
                    Vector clusterCenter = value.get();
                    centers.put(index, clusterCenter);
                }
            }
        }

        return centers;
    }

    private static List<Path> listOutputFiles(FileSystem fs, Path inputPath) throws IOException {
        FileStatus[] fileStatus = fs.listStatus(inputPath, new PathFilter() {
            @Override
            public boolean accept(Path path) {
                String pathName = path.getName();
                if (pathName.endsWith(".seq"))
                    return true;
                return path.getName().matches("part(.*)");
            }
        });
        List<Path> paths = Lists.newArrayList();
        for (FileStatus file : fileStatus) {
            paths.add(file.getPath());
        }
        return paths;
    }

    private MRUtils() {
    }
}
