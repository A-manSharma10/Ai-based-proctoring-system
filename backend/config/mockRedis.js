const logger = require('../utils/logger');

let mockStorage = {};

async function initializeRedis() {
    logger.info('Using MOCK Redis for demonstration');
    return Promise.resolve();
}

async function setSession(key, value, expireInSeconds = 3600) {
    mockStorage[key] = { value, expires: Date.now() + expireInSeconds * 1000 };
}

async function getSession(key) {
    const item = mockStorage[key];
    if (item && item.expires > Date.now()) {
        return item.value;
    }
    return null;
}

async function deleteSession(key) {
    delete mockStorage[key];
}

async function setCache(key, value, expireInSeconds = 300) {
    mockStorage[key] = { value, expires: Date.now() + expireInSeconds * 1000 };
}

async function getCache(key) {
    return getSession(key);
}

module.exports = {
    initializeRedis,
    setSession,
    getSession,
    deleteSession,
    setCache,
    getCache,
    client: { connect: () => Promise.resolve(), on: () => { } }
};
