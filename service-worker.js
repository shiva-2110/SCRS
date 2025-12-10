// Name of the cache
const CACHE_NAME = "cropx-cache-v1";

// Files to cache (add your login/register pages, CSS, JS)
const urlsToCache = [
  "/",
  "/login.html",
  "/register.html",
  "/static/images/icon-192.png",
  "/static/images/icon-512.png",
  "/static/styles.css"
];

// Install event: cache files
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

// Fetch event: serve cached files if available
self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});

// Activate event: cleanup old caches
self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(name => name !== CACHE_NAME)
                  .map(name => caches.delete(name))
      );
    })
  );
});
