const mysql = require('mysql2/promise');
const logger = require('../utils/logger');

if (process.env.USE_MOCK === 'true') {
  module.exports = require('./mockDb');
  return;
}

let pool;

const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  user: process.env.DB_USER || 'examuser',
  password: process.env.DB_PASSWORD || 'exampass',
  database: process.env.DB_NAME || 'exam_proctoring',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  acquireTimeout: 60000,
  timeout: 60000
};

async function initializeDatabase() {
  try {
    pool = mysql.createPool(dbConfig);

    // Test connection
    const connection = await pool.getConnection();
    await connection.ping();
    connection.release();

    logger.info('Database connected successfully');
  } catch (error) {
    logger.error('Database connection failed:', error);
    throw error;
  }
}

async function query(sql, params = []) {
  try {
    const [results] = await pool.execute(sql, params);
    return results;
  } catch (error) {
    logger.error('Database query error:', error);
    throw error;
  }
}

async function transaction(callback) {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const result = await callback(connection);
    await connection.commit();
    return result;
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

module.exports = {
  initializeDatabase,
  query,
  transaction,
  pool
};