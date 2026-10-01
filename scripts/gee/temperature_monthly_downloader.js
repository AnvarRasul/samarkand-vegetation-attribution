// ======================================================
// MOD11A2 MONTHLY LST DOWNLOAD SCRIPT (2015-2019)
// GOOGLE EARTH ENGINE JAVASCRIPT
// ======================================================

// ======================================================
// AOI
// ======================================================
// IMPORT YOUR SHAPEFILE AS AN ASSET:
//
// Assets -> NEW -> Shape files
//
// Then replace the path below
// ======================================================

var aoi = ee.FeatureCollection(
    'users/your_username/your_aoi'
);

// ======================================================
// SETTINGS
// ======================================================

var startYear = 2015;
var endYear = 2019;

// ======================================================
// MAP VIEW
// ======================================================

Map.centerObject(aoi, 7);
Map.addLayer(aoi, {}, 'AOI');

// ======================================================
// LOAD MOD11A2
// ======================================================

var mod11 = ee.ImageCollection('MODIS/061/MOD11A2')
    .select('LST_Day_1km');

// ======================================================
// FUNCTION:
// MONTHLY MEAN LST
// Kelvin -> Celsius
// ======================================================

function monthlyLST(year, month) {

    var start = ee.Date.fromYMD(year, month, 1);
    var end = start.advance(1, 'month');

    var collection = mod11
        .filterDate(start, end);

    // Monthly mean
    var meanLST = collection.mean();

    // Scale factor:
    // Kelvin = DN * 0.02
    // Celsius = Kelvin - 273.15

    var lstCelsius = meanLST
        .multiply(0.02)
        .subtract(273.15)
        .rename('LST_C');

    // Clip to AOI
    lstCelsius = lstCelsius.clip(aoi);

    // Reproject to EPSG:4326
    lstCelsius = lstCelsius.reproject({
        crs: 'EPSG:4326',
        scale: 1000
    });

    return lstCelsius.set({
        'year': year,
        'month': month
    });
}

// ======================================================
// VISUALIZATION
// ======================================================

var vis = {
    min: 0,
    max: 40,
};

var preview = monthlyLST(2019, 7);

Map.addLayer(preview, vis, 'Preview LST');

// ======================================================
// EXPORT LOOP
// ======================================================

for (var year = startYear; year <= endYear; year++) {

    for (var month = 1; month <= 12; month++) {

        var image = monthlyLST(year, month);

        var monthString = month < 10
            ? '0' + month
            : month.toString();

        var fileName =
            'MOD11A2_LST_' +
            year +
            '_' +
            monthString +
            '_samarkand_4326';

        Export.image.toDrive({

            image: image,

            description: fileName,

            folder: 'MOD11A2_MONTHLY',

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
// MOD11A2_LST_2015_01_samarkand_4326.tif
// MOD11A2_LST_2015_02_samarkand_4326.tif
// ...
// MOD11A2_LST_2019_12_samarkand_4326.tif
//
// ======================================================