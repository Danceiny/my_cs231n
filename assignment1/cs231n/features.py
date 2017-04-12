import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
from classifiers.linear_classifier import *
from classifiers import KNearestNeighbor

def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
        take as input an H x W x D array and return a (one-dimensional) array of
        length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in xrange(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print 'Done extracting features for %d / %d images' % (i, num_images)

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

        Parameters:
            rgb : RGB image

        Returns:
            gray : grayscale image
    
    """
    
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def gray_histogram(im, nbin=255, xmin=0, xmax=255, normed=True):
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)
    imhist, bin_edges = np.histogram(im, nbin, (xmin, xmax), normed)
    return imhist

def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image
    
             Modified from skimage.feature.hog
             http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
         
         Reference:
             Histograms of Oriented Gradients for Human Detection
             Navneet Dalal and Bill Triggs, CVPR 2005
         
        Parameters:
            im : an input grayscale or rgb image
            
        Returns:
            feat: Histogram of Gradient (HOG) feature
        
    """
    
    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)

    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape # image size
    orientations = 9 # number of gradient bins
    cx, cy = (8, 8) # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1) # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0) # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                                                grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                                                temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
    
    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
        1D vector of length nbin giving the color histogram over the hue of the
        input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram

    return imhist

def feat_test(train, val, test, feat_fns, classifiers):

    X_train,y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    for feature in feat_fns:   
        
        print "---------START-----------",feature[0].__name__,"--------START--------"
        X_train_feats = extract_features(X_train, feature, verbose=True)
        X_val_feats = extract_features(X_val, feature)
        X_test_feats = extract_features(X_test, feature)

        # Preprocessing: Subtract the mean feature
        mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
        X_train_feats -= mean_feat
        X_val_feats -= mean_feat
        X_test_feats -= mean_feat

        # Preprocessing: Divide by standard deviation. This ensures that each feature
        # has roughly the same scale.
        std_feat = np.std(X_train_feats, axis=0, keepdims=True)
        X_train_feats /= std_feat
        X_val_feats /= std_feat
        X_test_feats /= std_feat

        # Preprocessing: Add a bias dimension
        X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
        X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
        X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
        
        for classifier in classifiers:
 
            if classifier == "svm":
                print "---------------svm classifier---------------------"
                # Use the validation set to tune the learning rate and regularization strength
                learning_rates = [1e-9, 1e-8,  5e-7, 1e-7, 5e-8]
                regularization_strengths = [1e4, 5e4, 1e5,5e5, 1e6,5e6, 1e7,5e7,1e8]
                results = {}
                best_val = -1
                best_svm = None
                for lr in learning_rates:
                    for reg_str in regularization_strengths:
                        svm = LinearSVM()
                        _ = svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg_str,
                                              num_iters=300, verbose=False)
                        
                        y_train_pred = svm.predict(X_train_feats.T)
                        accuracy_train = np.mean(y_train == y_train_pred)
                        
                        y_val_pred = svm.predict(X_val_feats.T)
                        accuracy_val = np.mean(y_val == y_val_pred)
                        

                        results[(lr, reg_str)] = (accuracy_train, accuracy_val) 
                        if accuracy_val > best_val:
                            print "lr:",lr,
                            print "reg:", reg_str,
                            best_val = accuracy_val
                            print "cur_best_val:",best_val
                            best_svm = svm
                # Print out results.
                for lr, reg in sorted(results):
                    train_accuracy, val_accuracy = results[(lr, reg)]
                    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                                lr, reg, train_accuracy, val_accuracy)           
                print 'best validation accuracy achieved during cross-validation: %f' % best_val  
                
                print "# Evaluate your trained SVM on the test set"
                y_test_pred = best_svm.predict(X_test_feats.T)
                test_accuracy = np.mean(y_test == y_test_pred)
                print test_accuracy
                
                print "---------------END svm classifier END---------------------"
                print 
                
            elif classifier == "softmax":
                print "---------------softmax classifier---------------------"
                # Use the validation set to tune the learning rate and regularization strength
                learning_rates = [1e-9, 1e-8,  5e-7, 1e-7, 5e-8]
                regularization_strengths = [1e4, 5e4, 1e5,5e5, 1e6,5e6, 1e7,5e7,1e8]
                results = {}
                best_val = -1
                best_softmax = None
                for lr in learning_rates:
                    for reg_str in regularization_strengths:
                        softmax = Softmax()
                        _ = softmax.train(X_train_feats, y_train, learning_rate=lr, reg=reg_str,
                                              num_iters=1500, verbose=False)
                        y_train_pred = softmax.predict(X_train_feats.T)
                        accuracy_train = np.mean(y_train == y_train_pred)
                        y_val_pred = softmax.predict(X_val_feats.T)
                        accuracy_val = np.mean(y_val == y_val_pred)
                        results[(lr, reg_str)] = (accuracy_train, accuracy_val) 
                        if accuracy_val > best_val:
                            print "lr:",lr,
                            print "reg:", reg_str,
                            best_val = accuracy_val
                            print "cur_best_val:",best_val
                            best_softmax = softmax
                # Print out results.
                for lr, reg in sorted(results):
                    train_accuracy, val_accuracy = results[(lr, reg)]
                    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                                lr, reg, train_accuracy, val_accuracy)
                print 'best validation accuracy achieved during cross-validation: %f' % best_val   
                
                print "# Evaluate your trained softmax on the test set"
                y_test_pred = best_softmax.predict(X_test_feats.T)
                test_accuracy = np.mean(y_test == y_test_pred)
                print test_accuracy
                print "---------------END softmax classifier END---------------------"
                print 
                
            elif classifier == "knn":
                print "---------------knn classifier---------------------"
                best_val = -1
                k_choices = [1, 3, 5, 8, 10, 12, 15]
                knn = KNearestNeighbor()
                k_to_accuracies = {}
                method="euclid"
                #for method in dist_methods:        
                for k in k_choices:    
                    k_to_accuracies.setdefault(k,[])   
                    
                    knn.train(X_train_feats,y_train)    
                    
                    y_train_pred = knn.predict(X_train_feats, k, method=method)
                    y_val_pred = knn.predict(X_val_feats,k,method=method)
                    accuracy_train = np.mean(y_train == y_train_pred)
                    accuracy_val = np.mean(y_val == y_val_pred)
                    print "train=",accuracy_train," val=",accuracy_val
                    k_to_accuracies[k].append([accuracy_train,accuracy_val]) 
                    
                    if accuracy_val > best_val:
                        best_val = accuracy_val
                        print "cur_best_val:",best_val
                        best_k = k
                        
                for k in sorted(k_to_accuracies):
                    for accuracy in k_to_accuracies[k]:
                        print 'k = %d, accuracy: train = %f val = %f' % (k, accuracy[0],accuracy[1])
                
                print "# Evaluate your knn with best_k on the test set"
                y_test_pred = knn.predict(X_test_feats, k, method=method)
                accuracy_test = np.mean(y_test == y_test_pred)
                print accuracy_test
            else:
                print "classifier not exist....."
                return
            
        print "---------END-----------",feature[0].__name__,"--------END--------"

