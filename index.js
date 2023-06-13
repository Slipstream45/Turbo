const DistributedCache = require('./master');

const cache = new DistributedCache(4); // Create cache with 4 worker threads

async function cacheUsageExample() {
  try {
    // Set a value in the cache
    await cache.set('1', 'John');
    await cache.set('2', 'Andrey')

    // Get a value from the cache
    const value = await cache.get('2');
    console.log(value); // Output: value1

    // Check if a key exists in the cache
    const exists = await cache.has('1');
    console.log(exists); // Output: true

    // Delete a key from the cache
    await cache.delete('1');

    // Check if the key still exists
    const existsAfterDelete = await cache.has('1');
    console.log(existsAfterDelete); // Output: false
  } catch (error) {
    console.error(error);
  } finally {
    // Shutdown the cache when done
    cache.shutdown();
  }
}

cacheUsageExample();
