// ======================================================
// MOD13Q1 MONTHLY NDVI DOWNLOAD SCRIPT (2015-2024)
// GOOGLE EARTH ENGINE JAVASCRIPT
// ======================================================

// ======================================================
// AOI
// ======================================================
// Upload your shapefile to GEE Assets:
//
// Assets -> NEW -> Shape files
//
// Replace the asset path below
// ======================================================

var aoi = ee.FeatureCollection(
    'projects/your-gee-project/assets/samarkand_region'
);

// ======================================================
// SETTINGS
// ======================================================

var startYear = 2015;
var endYear = 2024;

// ======================================================
// MAP VIEW
// ======================================================

Map.centerObject(aoi, 7);
Map.addLayer(aoi, {}, 'AOI');

// ======================================================
// LOAD MOD13Q1 NDVI COLLECTION
// ======================================================

var mod13 = ee.ImageCollection('MODIS/061/MOD13Q1')
    .select('NDVI');

// ======================================================
// FUNCTION:
// MONTHLY MEAN NDVI
// ======================================================

function monthlyNDVI(year, month) {

    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');

    var collection = mod13
        .filterDate(start, end);

    // Monthly mean
    var meanNDVI = collection.mean();

    // MOD13Q1 scale factor:
    // NDVI = DN * 0.0001

    var ndvi = meanNDVI
        .multiply(0.0001)
        .rename('NDVI');

    // Clip to AOI
    ndvi = ndvi.clip(aoi);

    // Reproject to EPSG:4326
    ndvi = ndvi.reproject({
        crs: 'EPSG:4326',
        scale: 1000
    });

    return ndvi.set({
        'year': year,
        'month': month
    });
}

// ======================================================
// VISUALIZATION
// ======================================================

var vis = {
    min: 0,
    max: 1,
    palette: [
        'brown',
        'yellow',
        'green'
    ]
};

var preview = monthlyNDVI(2024, 7);

Map.addLayer(preview, vis, 'Preview NDVI');

// ======================================================
// EXPORT LOOP
// ======================================================

for (var year = startYear; year <= endYear; year++) {

    for (var month = 1; month <= 12; month++) {

        var image = monthlyNDVI(year, month);

        var monthString = month < 10
            ? '0' + month
            : month.toString();

        var fileName =
            'MOD13Q1_NDVI_' +
            year +
            '_' +
            monthString +
            '_samarkand_4326';

        Export.image.toDrive({

            image: image,

            description: fileName,

            folder: 'MOD13Q1_MONTHLY_NDVI',

            fileNamePrefix: fileName,

            region: aoi.geometry(),

            scale: 1000,

            crs: 'EPSG:4326',

            maxPixels: 1e13
        });
    }
}

// ======================================================
// DONE
// ======================================================
//
// OUTPUT:
//
// MOD13Q1_NDVI_2015_01_samarkand_4326.tif
// MOD13Q1_NDVI_2015_02_samarkand_4326.tif
// ...
// MOD13Q1_NDVI_2024_12_samarkand_4326.tif
//
// ======================================================