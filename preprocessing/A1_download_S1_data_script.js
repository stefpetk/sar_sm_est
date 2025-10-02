// saved an imported shapefile and a Sentinel-1 GRD Image collection
// as a variable
// clip the Sentinel-1 GRD Images according to the area of interest shapefile and
// then retain only the images that were taken during the dates of interest
var area_of_interest = table;
var s1_GRD_dataset = ee.FeatureCollection(imageCollection)
  .filterBounds(area_of_interest)
  .filterDate('2016-01-01', '2016-12-31')
  .filterMetadata('relativeOrbitNumber_start', 'equals', 63);
  
// clip the images and then export them to google drive
var batch = require('users/fitoprincipe/geetools:batch');
var s1_GRD_images = ee.ImageCollection(s1_GRD_dataset);
var s1_GRD_im_clip = s1_GRD_images.map(function (image) {
  return image.clip(area_of_interest);
});

var im_list = s1_GRD_im_clip.toList(s1_GRD_im_clip.size());
var n = im_list.size().getInfo();
for (var i = 0; i < n; i++) {
  var list_of_im = s1_GRD_im_clip.toList(s1_GRD_im_clip.size());
  var list_of_im_el = list_of_im.get(i);
  var all_images = ee.ImageCollection.fromImages([list_of_im_el]);
  
  batch.Download.ImageCollection.toDrive(all_images, "S1 GRD Images (STATION_NAME)", 
      {name: '{system:index}',
      scale: 10,
      maxPixels: 10e09,
      region: area_of_interest
      });
}
