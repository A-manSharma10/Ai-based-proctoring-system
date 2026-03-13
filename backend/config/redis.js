const redis = require('redis');
const logger = require('../utils/logger');

if (process.env.USE_MOCK === 'true') {
  module.exports = require('./mockRedis');
  return;
}

let client;

async function initializeRedis() {
  try {
    client = redis.createClient({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      retry_strategy: (options) => {
        if (options.error && options.error.code === 'ECONNREFUSED') {
          logger.error('Redis server connection refused');
          return new Error('Redis server connection refused');
        }
        if (options.total_retry_time > 1000 * 60 * 60) {
          return new Error('Redis retry time exhausted');
        }
        if (options.attempt > 10) {
          return undefined;
        }
        return Math.min(options.attempt * 100, 3000);
      }
    });

    client.on('error', (err) => {
      logger.error('Redis client error:', err);
    });

    client.on('connect', () => {
      logger.info('Redis client connected');
    });

    await client.connect();
    logger.info('Redis initialized successfully');
  } catch (error) {
    logger.error('Redis initialization failed:', error);
    throw error;
  }
}

async function setSession(key, value, expireInSeconds = 3600) {
  try {
    await client.setEx(key, expireInSeconds, JSON.stringify(value));
  } catch (error) {
    logger.error('Redis set error:', error);
    throw error;
  }
}

async function getSession(key) {
  try {
    const value = await client.get(key);
    return value ? JSON.parse(value) : null;
  } catch (error) {
    logger.error('Redis get error:', error);
    throw error;
  }
}

async function deleteSession(key) {
  try {
    await client.del(key);
  } catch (error) {
    logger.error('Redis delete error:', error);
    throw error;
  }
}

async function setCache(key, value, expireInSeconds = 300) {
  try {
    await client.setEx(key, expireInSeconds, JSON.stringify(value));
  } catch (error) {
    logger.error('Redis cache set error:', error);
    throw error;
  }
}

async function getCache(key) {
  try {
    const value = await client.get(key);
    return value ? JSON.parse(value) : null;
  } catch (error) {
    logger.error('Redis cache get error:', error);
    throw error;
  }
}

module.exports = {
  initializeRedis,
  setSession,
  getSession,
  deleteSession,
  setCache,
  getCache,
  client
};