const axios = require('axios');

const services = [
  { name: 'Frontend', url: 'http://localhost:3000' },
  { name: 'Backend API', url: 'http://localhost:5000/health' },
  { name: 'Face Recognition Service', url: 'http://localhost:8001/health' },
  { name: 'Object Detection Service', url: 'http://localhost:8002/health' },
  { name: 'Audio Analysis Service', url: 'http://localhost:8003/health' },
  { name: 'Behavioral Analysis Service', url: 'http://localhost:8004/health' }
];

async function checkService(service) {
  try {
    const response = await axios.get(service.url, { timeout: 5000 });
    return {
      name: service.name,
      status: 'healthy',
      statusCode: response.status,
      responseTime: response.headers['x-response-time'] || 'N/A'
    };
  } catch (error) {
    return {
      name: service.name,
      status: 'unhealthy',
      error: error.message,
      statusCode: error.response?.status || 'N/A'
    };
  }
}

async function healthCheck() {
  console.log('🏥 AI Exam Proctoring System Health Check\n');
  console.log('Checking all services...\n');

  const results = await Promise.all(services.map(checkService));
  
  let healthyCount = 0;
  let totalCount = results.length;

  results.forEach(result => {
    const statusIcon = result.status === 'healthy' ? '✅' : '❌';
    const statusText = result.status === 'healthy' ? 'HEALTHY' : 'UNHEALTHY';
    
    console.log(`${statusIcon} ${result.name}: ${statusText}`);
    
    if (result.status === 'healthy') {
      console.log(`   Status Code: ${result.statusCode}`);
      if (result.responseTime !== 'N/A') {
        console.log(`   Response Time: ${result.responseTime}`);
      }
      healthyCount++;
    } else {
      console.log(`   Error: ${result.error}`);
      console.log(`   Status Code: ${result.statusCode}`);
    }
    console.log('');
  });

  console.log(`📊 Overall Health: ${healthyCount}/${totalCount} services healthy`);
  
  if (healthyCount === totalCount) {
    console.log('🎉 All services are running properly!');
    process.exit(0);
  } else {
    console.log('⚠️  Some services are not responding. Check the logs for more details.');
    console.log('   Run: docker-compose logs -f');
    process.exit(1);
  }
}

// Run health check
healthCheck().catch(error => {
  console.error('❌ Health check failed:', error.message);
  process.exit(1);
});