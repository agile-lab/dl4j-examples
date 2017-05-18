/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.examples.deeplogo;


import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.collection.CompactHeapStringList;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;

/**
 * Image record reader.
 * Reads a local file system and parses images of a given
 * height and width.
 * All images are rescaled and converted to the given height, width, and number of channels.
 *
 * Also appends the label if specified
 * (one of k encoding based on the directory structure where each subdir of the root is an indexed label)
 * @author Adam Gibson
 */
public class MyImageRecordReader extends BaseImageRecordReader {


    /** Loads images with height = 28, width = 28, and channels = 1, appending no labels. */
    public MyImageRecordReader() {
        super();
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator. */
    public MyImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator) {
        super(height, width, channels, labelGenerator);
        cropImage = true;
    }

    /** Loads images with given height, width, and channels, appending no labels. */
    public MyImageRecordReader(int height, int width, int channels) {
        super(height, width, channels, (PathLabelGenerator) null);
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator. */
    public MyImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator,
                    ImageTransform imageTransform) {
        super(height, width, channels, labelGenerator, imageTransform);
    }

    /** Loads images with given height, width, and channels, appending no labels. */
    public MyImageRecordReader(int height, int width, int channels, ImageTransform imageTransform) {
        super(height, width, channels, null, imageTransform);
    }

    /** Loads images with given  height, width, and channels, appending labels returned by the generator. */
    public MyImageRecordReader(int height, int width, PathLabelGenerator labelGenerator) {
        super(height, width, 1, labelGenerator);
    }

    /** Loads images with given height, width, and channels = 1, appending no labels. */
    public MyImageRecordReader(int height, int width) {
        super(height, width, 1, null, null);
    }


    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new MyNativeImageLoader(height, width, channels, imageTransform, cropImage);
        }
        inputSplit = split;
        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (locations.length > 1 || containsFormat(locations[0].getPath())) {
                allPaths = new CompactHeapStringList();
                for (URI location : locations) {
                    File imgFile = new File(location);
                    if (!imgFile.isDirectory() && containsFormat(imgFile.getAbsolutePath())) {
                        allPaths.add(imgFile.toURI().toString());
                    }
                    if (appendLabel) {
                        File parentDir = imgFile.getParentFile();
                        String name = parentDir.getName();
                        if (labelGenerator != null) {
                            name = labelGenerator.getLabelForPath(location).toString();
                        }
                        if (!labels.contains(name)) {
                            labels.add(name);
                        }
                        if (pattern != null) {
                            String label = name.split(pattern)[patternPosition];
                            fileNameMap.put(imgFile.toString(), label);
                        }
                    }
                }
            } else {
                File curr = new File(locations[0]);
                if (!curr.exists())
                    throw new IllegalArgumentException("Path " + curr.getAbsolutePath() + " does not exist!");
                if (curr.isDirectory()) {
                    Collection<File> temp = FileUtils.listFiles(curr, null, true);
                    allPaths = new CompactHeapStringList();
                    for (File f : temp) {
                        allPaths.add(f.getPath());
                    }
                } else {
                    allPaths = Collections.singletonList(curr.getPath());
                }

            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
        }
        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        Collections.sort(labels);
    }

}
