const { Octokit } = require('@octokit/rest');
require('dotenv').config();

// 解决Windows环境下的中文乱码问题
if (process.platform === 'win32') {
  // 设置控制台编码为UTF-8
  try {
    const cp = require('child_process');
    cp.execSync('chcp 65001', { stdio: 'ignore' });
  } catch (e) {
    // 忽略错误
  }
}

// 设置输出编码为utf8
process.stdout.setEncoding('utf8');
process.stderr.setEncoding('utf8');

class WorkflowTestAnalyzer {
  constructor(workflowName = null, runId = null) {
    const octokitOptions = {
      auth: process.env.GITHUB_TOKEN
    };
    
    // 处理证书验证问题
    if (process.env.NODE_TLS_REJECT_UNAUTHORIZED === '0') {
      octokitOptions.request = {
        agent: new (require('https').Agent)({ rejectUnauthorized: false })
      };
    }
    
    this.octokit = new Octokit(octokitOptions);
    this.owner = process.env.GITHUB_OWNER;
    this.repo = process.env.GITHUB_REPO;
    this.branch = process.env.GITHUB_BRANCH;
    this.workflowName = workflowName;
    this.runId = runId;
    this.debug = process.env.DEBUG === 'true';
  }

  async getWorkflowRuns(timeout = 15000) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      
      const requestOptions = {
        owner: this.owner,
        repo: this.repo,
        per_page: 20, // 临时增加获取更多工作流运行
        request: {
          signal: controller.signal
        }
      };
      
      // 如果指定了分支，添加branch参数
      if (this.branch) {
        requestOptions.branch = this.branch;
      }
      
      const response = await this.octokit.actions.listWorkflowRunsForRepo(requestOptions);
      
      clearTimeout(timeoutId);
      
      let workflowRuns = response.data.workflow_runs;
      
      // 打印所有工作流运行，以便找到包含full-1-npu-a3的运行
      console.log('=== All Workflow Runs ===');
      workflowRuns.forEach(run => {
        console.log(`ID: ${run.id}, Name: ${run.name}, Status: ${run.status}`);
      });
      console.log('====================');
      
      // Filter by workflow name if specified
      if (this.workflowName) {
        workflowRuns = workflowRuns.filter(run => run.name === this.workflowName);
      }
      
      return workflowRuns;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('Error fetching workflow runs: Request timed out');
      } else {
        console.error('Error fetching workflow runs:', error.message);
      }
      return [];
    }
  }

  async getWorkflowRunJobs(runId, timeout = 15000) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      
      const response = await this.octokit.actions.listJobsForWorkflowRun({
        owner: this.owner,
        repo: this.repo,
        run_id: runId,
        request: {
          signal: controller.signal
        }
      });
      
      clearTimeout(timeoutId);
      return response.data.jobs;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error(`Error fetching jobs for run ${runId}: Request timed out`);
      } else {
        console.error(`Error fetching jobs for run ${runId}:`, error.message);
      }
      return [];
    }
  }

  async getJobLogs(jobId, timeout = 30000, maxRetries = 3) {
    let retries = 0;
    
    // 初始等待时间，让日志有足够时间同步
    console.log(`    Waiting for logs to sync...`);
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    while (retries <= maxRetries) {
      try {
        // 添加超时处理
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        console.log(`    Fetching logs (attempt ${retries + 1}/${maxRetries + 1})...`);
        const response = await this.octokit.actions.downloadJobLogsForWorkflowRun({
          owner: this.owner,
          repo: this.repo,
          job_id: jobId,
          request: {
            signal: controller.signal
          }
        });
        
        clearTimeout(timeoutId);
        
        // 检查响应数据是否为空
        if (!response.data || response.data.length === 0) {
          console.log(`    API returned empty data`);
          if (retries < maxRetries) {
            retries++;
            const waitTime = 2000 * retries; // 递增等待时间
            console.log(`    Empty response, retrying (${retries}/${maxRetries}) in ${waitTime}ms...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          } else {
            console.log(`    No data received after ${maxRetries} retries`);
            return null;
          }
        }
        
        // 将日志保存到本地文件
        const fs = require('fs');
        const logFilePath = `./logs/job_${jobId}_logs.txt`;
        
        // 确保logs目录存在
        if (!fs.existsSync('./logs')) {
          fs.mkdirSync('./logs', { recursive: true });
        }
        
        // 保存日志到文件
        fs.writeFileSync(logFilePath, response.data);
        
        // 检查文件是否为空
        const stats = fs.statSync(logFilePath);
        console.log(`    Log file size: ${stats.size} bytes`);
        
        if (stats.size === 0) {
          if (retries < maxRetries) {
            retries++;
            const waitTime = 2000 * retries; // 递增等待时间
            console.log(`    Log file is empty, retrying (${retries}/${maxRetries}) in ${waitTime}ms...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            continue;
          } else {
            console.log(`    Logs saved to: ${logFilePath} (empty after ${maxRetries} retries)`);
            return logFilePath;
          }
        }
        
        console.log(`    Logs saved to: ${logFilePath} (${stats.size} bytes)`);
        return logFilePath;
      } catch (error) {
        if (error.name === 'AbortError') {
          console.error(`Error fetching logs for job ${jobId}: Request timed out`);
        } else {
          console.error(`Error fetching logs for job ${jobId}:`, error.message);
        }
        
        if (retries < maxRetries) {
          retries++;
          const waitTime = 2000 * retries; // 递增等待时间
          console.log(`    Error fetching logs, retrying (${retries}/${maxRetries}) in ${waitTime}ms...`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        } else {
          return null;
        }
      }
    }
    return null;
  }

  async analyzeJobDirectly(jobId) {
    console.log(`Analyzing job #${jobId} for ${this.owner}/${this.repo}...`);
    
    try {
      // 获取job信息
      const jobInfo = await this.getJobInfo(jobId);
      const logFilePath = await this.getJobLogs(jobId);
      
      if (logFilePath) {
        const testResults = this.analyzeTestResults(logFilePath);
        
        // 生成MD格式报告
        const mdReport = await this.generateMdReport(jobInfo, testResults);
        
        // 保存MD报告到文件
        const fs = require('fs');
        const reportDir = './reports';
        if (!fs.existsSync(reportDir)) {
          fs.mkdirSync(reportDir, { recursive: true });
        }
        
        const reportFileName = `test_report_job_${jobId}_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
        const reportFilePath = `${reportDir}/${reportFileName}`;
        fs.writeFileSync(reportFilePath, mdReport);
        
        console.log(`\n=== Job Analysis Results ===`);
        console.log(`Test Results:`);
        console.log(`  Total: ${testResults.total}`);
        console.log(`  Passed: ${testResults.passed}`);
        console.log(`  Failed: ${testResults.failed}`);
        console.log(`  Skipped: ${testResults.skipped}`);

        if (testResults.failures.length > 0) {
          console.log(`Failed Tests:`);
          testResults.failures.forEach((failure, index) => {
            console.log(`  ${index + 1}. ${failure}`);
          });
        }
        
        if (testResults.passedTests.length > 0) {
          console.log(`Passed Tests:`);
          // 只显示前5个通过的测试，避免输出过多
          const displayedPassedTests = testResults.passedTests.slice(0, 5);
          displayedPassedTests.forEach((test, index) => {
            console.log(`  ${index + 1}. ${test}`);
          });
          if (testResults.passedTests.length > 5) {
            console.log(`  ... and ${testResults.passedTests.length - 5} more passed tests`);
          }
        }
        
        console.log(`\nMD report generated: ${reportFilePath}`);
      }
    } catch (error) {
      console.error(`Error analyzing job:`, error.message);
    }
  }

  async getJobInfo(jobId) {
    try {
      const response = await this.octokit.actions.getJobForWorkflowRun({
        owner: this.owner,
        repo: this.repo,
        job_id: jobId
      });
      return response.data;
    } catch (error) {
      console.error(`Error fetching job info:`, error.message);
      return null;
    }
  }

  async getLatestCommitter(filePath) {
    try {
      const requestOptions = {
        owner: this.owner,
        repo: this.repo,
        path: filePath,
        per_page: 1
      };
      
      // 如果指定了分支，添加分支参数
      if (this.branch) {
        requestOptions.sha = this.branch;
      }
      
      const response = await this.octokit.repos.listCommits(requestOptions);
      if (response.data.length > 0) {
        return response.data[0].author?.login || '';
      }
      return '';
    } catch (error) {
      console.error(`Error fetching latest committer for ${filePath}:`, error.message);
      return '';
    }
  }

  async createGitHubIssue(workflowRun, allTestResults) {
    try {
      // 生成完整的MD报告
      const mdReport = await this.generateWorkflowMdReport(workflowRun, allTestResults);
      
      // 准备issue标题（英文格式）
      const workflowName = workflowRun.name;
      const createdTime = new Date(workflowRun.created_at).toLocaleString();
      const conclusion = workflowRun.conclusion;
      const issueTitle = `[TestReport] ${workflowName} was executed at ${createdTime}, result: ${conclusion}`;
      
      // 准备issue内容（包含完整MD报告）
      let issueBody = mdReport;
      
      // 创建issue
      const response = await this.octokit.issues.create({
        owner: this.owner,
        repo: this.repo,
        title: issueTitle,
        body: issueBody,
        labels: ['bug']
      });
      
      console.log(`\nGitHub issue created: ${response.data.html_url}`);
      return response.data;
    } catch (error) {
      console.error(`Error creating GitHub issue:`, error.message);
      return null;
    }
  }

  async generateMdReport(jobInfo, testResults) {
    let md = `# Test Report\n\n`;
    
    // 添加执行信息
    md += `## Execution Information\n`;
    md += `- **Job ID**: ${jobInfo ? jobInfo.id : 'N/A'}\n`;
    md += `- **Job Name**: ${jobInfo ? jobInfo.name : 'N/A'}\n`;
    md += `- **Status**: ${jobInfo ? jobInfo.status : 'N/A'}\n`;
    md += `- **Conclusion**: ${jobInfo ? jobInfo.conclusion : 'N/A'}\n`;
    md += `- **Analysis Date**: ${new Date().toISOString()}\n\n`;
    
    // 添加Job执行统计结果表格
    md += `## Job Execution Statistics\n`;
    md += `| 序号 | Job Name | Status | Conclusion | Total | Passed | Failed | Skipped |\n`;
    md += `|------|----------|--------|------------|-------|--------|--------|---------|\n`;
    md += `| 1 | ${jobInfo ? jobInfo.name : 'N/A'} | ${jobInfo ? jobInfo.status : 'N/A'} | ${jobInfo ? jobInfo.conclusion : 'N/A'} | ${testResults.total} | ${testResults.passed} | ${testResults.failed} | ${testResults.skipped} |\n\n`;
    
    // 添加失败用例详情表格
    if (testResults.failures.length > 0) {
      md += `## Failed Test Details\n`;
      md += `| 序号 | Test Case Path | Job Name | Analysis Owner | Analysis Result |\n`;
      md += `|------|---------------|----------|----------------|-----------------|\n`;
      
      let failureIndex = 1;
      for (const failure of testResults.failures) {
        // 提取测试用例路径
        let testPath = failure;
        let relativePath = failure;
        if (failure.includes('/__w/')) {
          const fullPath = failure.match(/\/__w\/[^\s]+\.py/)[0];
          // 只保留文件名部分用于显示
          testPath = fullPath.split('/').pop();
          
          // 提取相对路径用于GitHub API查询
          if (fullPath.includes('/__w/sglang/sglang/')) {
            relativePath = fullPath.replace('/__w/sglang/sglang/', '');
          } else {
            // 尝试提取通用的相对路径
            const pathParts = fullPath.split('/__w/');
            if (pathParts.length > 1) {
              const repoPath = pathParts[1];
              const repoNameEndIndex = repoPath.indexOf('/');
              if (repoNameEndIndex > 0) {
                relativePath = repoPath.substring(repoNameEndIndex + 1);
              }
            }
          }
        }
        
        // 查询最新修改人
        let analysisOwner = '';
        try {
          analysisOwner = await this.getLatestCommitter(relativePath);
        } catch (error) {
          console.error(`Error getting latest committer:`, error.message);
        }
        
        md += `| ${failureIndex++} | ${testPath} | ${jobInfo ? jobInfo.name : 'N/A'} | ${analysisOwner} | |\n`;
      }
      md += `\n`;
    }
    
    return md;
  }

  async getWorkflowRunById(runId, timeout = 15000) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      
      const response = await this.octokit.actions.getWorkflowRun({
        owner: this.owner,
        repo: this.repo,
        run_id: runId,
        request: {
          signal: controller.signal
        }
      });
      
      clearTimeout(timeoutId);
      return response.data;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error(`Error fetching workflow run ${runId}: Request timed out`);
      } else {
        console.error(`Error fetching workflow run ${runId}:`, error.message);
      }
      return null;
    }
  }

  analyzeTestResults(logFilePath) {
    const testResults = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      failures: [],
      passedTests: []
    };

    try {
      // 读取本地日志文件
      const fs = require('fs');
      
      // 检查文件是否为空
      const stats = fs.statSync(logFilePath);
      if (stats.size === 0) {
        console.error(`Error analyzing test results: Log file is empty - ${logFilePath}`);
        return testResults;
      }
      
      const logs = fs.readFileSync(logFilePath, 'utf8');
      const lines = logs.split('\n');
      
      let inPassedSection = false;
      let inFailedSection = false;
      
      for (const line of lines) {
        const cleanedLine = line.trim();
        
        // 检查PASSED章节
        if (cleanedLine.includes('✓ PASSED:')) {
          inPassedSection = true;
          inFailedSection = false;
        }
        
        // 检查FAILED章节
        else if (cleanedLine.includes('✗ FAILED:')) {
          inPassedSection = false;
          inFailedSection = true;
        }
        
        // 检查章节结束
        else if (cleanedLine.includes('============================================================')) {
          inPassedSection = false;
          inFailedSection = false;
        }
        
        // 处理PASSED章节中的测试用例
        else if (inPassedSection && cleanedLine.includes('.py')) {
          testResults.passedTests.push(cleanedLine);
        }
        
        // 处理FAILED章节中的测试用例
        else if (inFailedSection) {
          // 只统计包含.py的测试用例，排除警告信息
          if (cleanedLine.includes('.py') && (cleanedLine.includes('exit code') || cleanedLine.includes('(exit code'))) {
            testResults.failures.push(cleanedLine);
          }
        }
      }
      
      // 通过PASSED和FAILED章节计算测试结果
      testResults.passed = testResults.passedTests.length;
      testResults.failed = testResults.failures.length;
      testResults.total = testResults.passed + testResults.failed;
      
    } catch (error) {
      console.error('Error analyzing test results:', error.message);
    }

    return testResults;
  }

  async analyzeWorkflowTests() {
    let workflowRun;
    
    if (this.runId) {
      if (this.branch) {
        console.log(`Analyzing workflow run #${this.runId} for ${this.owner}/${this.repo} (branch: ${this.branch})...`);
      } else {
        console.log(`Analyzing workflow run #${this.runId} for ${this.owner}/${this.repo}...`);
      }
      workflowRun = await this.getWorkflowRunById(this.runId);
      
      if (!workflowRun) {
        console.log(`Workflow run #${this.runId} not found`);
        return;
      }
    } else {
      if (this.workflowName) {
        if (this.branch) {
          console.log(`Analyzing latest workflow "${this.workflowName}" test results for ${this.owner}/${this.repo} (branch: ${this.branch})...`);
        } else {
          console.log(`Analyzing latest workflow "${this.workflowName}" test results for ${this.owner}/${this.repo}...`);
        }
      } else {
        if (this.branch) {
          console.log(`Analyzing latest workflow test results for ${this.owner}/${this.repo} (branch: ${this.branch})...`);
        } else {
          console.log(`Analyzing latest workflow test results for ${this.owner}/${this.repo}...`);
        }
      }
      
      const workflowRuns = await this.getWorkflowRuns();
      
      if (workflowRuns.length === 0) {
        if (this.workflowName) {
          console.log(`No workflow runs found for "${this.workflowName}"`);
        } else {
          console.log('No workflow runs found');
        }
        return;
      }
      
      // 查找第一个已完成的工作流运行
      workflowRun = workflowRuns.find(run => run.status === 'completed');
      
      if (!workflowRun) {
        // 如果没有找到已完成的运行，尝试获取更多运行记录
        const moreRequestOptions = {
          owner: this.owner,
          repo: this.repo,
          per_page: 20,
        };
        
        if (this.branch) {
          moreRequestOptions.branch = this.branch;
        }
        
        const moreResponse = await this.octokit.actions.listWorkflowRunsForRepo(moreRequestOptions);
        const moreWorkflowRuns = moreResponse.data.workflow_runs;
        
        // 如果指定了工作流名称，过滤出对应名称的工作流
        const filteredRuns = this.workflowName 
          ? moreWorkflowRuns.filter(run => run.name === this.workflowName)
          : moreWorkflowRuns;
        
        // 查找第一个已完成的运行
        workflowRun = filteredRuns.find(run => run.status === 'completed');
        
        if (!workflowRun) {
          if (this.workflowName) {
            console.log(`No completed workflow runs found for "${this.workflowName}"`);
          } else {
            console.log('No completed workflow runs found');
          }
          return;
        }
        
        console.log(`Found completed workflow run: #${workflowRun.id} (${workflowRun.name})`);
      } else {
        console.log(`Found latest workflow run: #${workflowRun.id} (${workflowRun.name})`);
      }
    }
    
    console.log(`\n=== Workflow: ${workflowRun.name} (ID: ${workflowRun.id}) ===`);
    console.log(`Status: ${workflowRun.status}`);
    console.log(`Conclusion: ${workflowRun.conclusion}`);
    console.log(`Event: ${workflowRun.event}`);
    console.log(`Created: ${new Date(workflowRun.created_at).toLocaleString()}`);
    
    const jobs = await this.getWorkflowRunJobs(workflowRun.id);
    console.log(`Jobs: ${jobs.length}`);

    // 收集所有job的测试结果，用于生成MD报告
    const allTestResults = [];

    for (const job of jobs) {
      console.log(`\n  - Job: ${job.name} (ID: ${job.id})`);
      console.log(`    Status: ${job.status}`);
      console.log(`    Conclusion: ${job.conclusion}`);

      // 统计分析作业名称中包含"-npu-"的任务
      if (job.name.toLowerCase().includes('-npu-')) {
        console.log(`    📋 Analyzing test job logs...`);
        try {
          const logFilePath = await this.getJobLogs(job.id);
          let testResults = {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: [],
            passedTests: []
          };
          
          if (logFilePath) {
            // 检查文件是否为空
            const fs = require('fs');
            const stats = fs.statSync(logFilePath);
            
            if (stats.size > 0) {
              testResults = this.analyzeTestResults(logFilePath);
              
              console.log(`    Test Results:`);
              console.log(`      Total: ${testResults.total}`);
              console.log(`      Passed: ${testResults.passed}`);
              console.log(`      Failed: ${testResults.failed}`);
              console.log(`      Skipped: ${testResults.skipped}`);

              if (testResults.failures.length > 0) {
                console.log(`    Failed Tests:`);
                testResults.failures.forEach((failure, index) => {
                  console.log(`      ${index + 1}. ${failure}`);
                });
              }
              
              if (testResults.passedTests.length > 0) {
                console.log(`    Passed Tests:`);
                // 只显示前5个通过的测试，避免输出过多
                const displayedPassedTests = testResults.passedTests.slice(0, 5);
                displayedPassedTests.forEach((test, index) => {
                  console.log(`      ${index + 1}. ${test}`);
                });
                if (testResults.passedTests.length > 5) {
                  console.log(`      ... and ${testResults.passedTests.length - 5} more passed tests`);
                }
              }
            } else {
              console.log(`    Test Results: Log file is empty`);
            }
          }
          
          // 无论日志文件是否为空，都添加到统计中
          allTestResults.push({ job, testResults });
        } catch (error) {
          console.error(`    Error analyzing test job:`, error.message);
          
          // 即使发生错误，也添加到统计中，使用默认值
          const testResults = {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            failures: [],
            passedTests: []
          };
          allTestResults.push({ job, testResults });
        }
      }
    }

    // 生成MD格式报告
    if (allTestResults.length > 0) {
      const mdReport = await this.generateWorkflowMdReport(workflowRun, allTestResults);
      
      // 保存MD报告到文件
      const fs = require('fs');
      const reportDir = './reports';
      if (!fs.existsSync(reportDir)) {
        fs.mkdirSync(reportDir, { recursive: true });
      }
      
      const reportFileName = `test_report_workflow_${workflowRun.id}_${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
      const reportFilePath = `${reportDir}/${reportFileName}`;
      fs.writeFileSync(reportFilePath, mdReport);
      
      console.log(`\nMD report generated: ${reportFilePath}`);
      
      // 不自动创建GitHub issue，用户可以通过--create-issue命令手动创建
      console.log('\nTo create a GitHub issue with this report, run:');
      console.log(`node analyze-workflow-tests.js --create-issue ${reportFilePath}`);
    }
  }

  async createIssueFromReport(reportPath) {
    try {
      const fs = require('fs');
      if (!fs.existsSync(reportPath)) {
        console.error(`Error: Report file not found - ${reportPath}`);
        return null;
      }
      
      const mdContent = fs.readFileSync(reportPath, 'utf8');
      
      // 从报告中提取信息
      const workflowNameMatch = mdContent.match(/\*\*Workflow Name\*\*: (.+)/);
      const createdMatch = mdContent.match(/\*\*Created\*\*: (.+)/);
      const conclusionMatch = mdContent.match(/\*\*Conclusion\*\*: (.+)/);
      
      const workflowName = workflowNameMatch ? workflowNameMatch[1] : 'Unknown Workflow';
      const createdTime = createdMatch ? createdMatch[1] : new Date().toLocaleString();
      const conclusion = conclusionMatch ? conclusionMatch[1] : 'Unknown';
      
      // 准备issue标题（英文格式）
      const issueTitle = `[TestReport] ${workflowName} was executed at ${createdTime}, result: ${conclusion}`;
      
      // 创建issue
      const response = await this.octokit.issues.create({
        owner: this.owner,
        repo: this.repo,
        title: issueTitle,
        body: mdContent,
        labels: ['bug']
      });
      
      console.log(`\nGitHub issue created: ${response.data.html_url}`);
      return response.data;
    } catch (error) {
      console.error(`Error creating GitHub issue from report:`, error.message);
      return null;
    }
  }

  async generateWorkflowMdReport(workflowRun, allTestResults) {
    let md = `# Test Report\n\n`;
    
    // 添加执行信息
    md += `## Execution Information\n`;
    md += `- **Workflow ID**: ${workflowRun.id}\n`;
    md += `- **Workflow Name**: ${workflowRun.name}\n`;
    md += `- **Status**: ${workflowRun.status}\n`;
    md += `- **Conclusion**: ${workflowRun.conclusion}\n`;
    md += `- **Event**: ${workflowRun.event}\n`;
    md += `- **Created**: ${new Date(workflowRun.created_at).toLocaleString()}\n`;
    md += `- **Analysis Date**: ${new Date().toISOString()}\n\n`;
    
    // 添加Job执行统计结果表格
    md += `## Job Execution Statistics\n`;
    md += `| Index | Job Name | Status | Conclusion | Total | Passed | Failed | Skipped |\n`;
    md += `|-------|----------------------------|--------|------------|-------|--------|--------|---------|\n`;
    
    // 计算汇总数据
    let totalTotal = 0;
    let totalPassed = 0;
    let totalFailed = 0;
    let totalSkipped = 0;
    let jobIndex = 1;
    
    allTestResults.forEach(({ job, testResults }) => {
      md += `| ${jobIndex++} | ${job.name} | ${job.status} | ${job.conclusion} | ${testResults.total} | ${testResults.passed} | ${testResults.failed} | ${testResults.skipped} |\n`;
      
      totalTotal += testResults.total;
      totalPassed += testResults.passed;
      totalFailed += testResults.failed;
      totalSkipped += testResults.skipped;
    });
    
    // 添加汇总行
    md += `| **总计** | **Total** | ${workflowRun.status} | ${workflowRun.conclusion} | ${totalTotal} | ${totalPassed} | ${totalFailed} | ${totalSkipped} |\n`;
    md += `\n`;
    
    // 添加失败用例详情表格
    const allFailures = allTestResults.flatMap(({ job, testResults }) => {
      return testResults.failures.map(failure => ({ job, failure }));
    });
    
    if (allFailures.length > 0) {
      md += `## Failed Test Details\n`;
      md += `| Index | Test Case Path | Job Name | Analysis Owner | Analysis Result |\n`;
      md += `|-------|----------------------------|----------------------------|----------------|-----------------|\n`;
      
      let failureIndex = 1;
      for (const { job, failure } of allFailures) {
        // 提取测试用例路径
        let testPath = failure;
        let relativePath = failure;
        if (failure.includes('/__w/')) {
          const fullPath = failure.match(/\/__w\/[^\s]+\.py/)[0];
          // 只保留文件名部分用于显示
          testPath = fullPath.split('/').pop();
          
          // 提取相对路径用于GitHub API查询
          if (fullPath.includes('/__w/sglang/sglang/')) {
            relativePath = fullPath.replace('/__w/sglang/sglang/', '');
          } else {
            // 尝试提取通用的相对路径
            const pathParts = fullPath.split('/__w/');
            if (pathParts.length > 1) {
              const repoPath = pathParts[1];
              const repoNameEndIndex = repoPath.indexOf('/');
              if (repoNameEndIndex > 0) {
                relativePath = repoPath.substring(repoNameEndIndex + 1);
              }
            }
          }
        }
        
        // 查询最新修改人
        let analysisOwner = '';
        try {
          analysisOwner = await this.getLatestCommitter(relativePath);
        } catch (error) {
          console.error(`Error getting latest committer:`, error.message);
        }
        
        md += `| ${failureIndex++} | ${testPath} | ${job.name} | ${analysisOwner} | |\n`;
      }
      md += `\n`;
    }
    
    return md;
  }
}

// 运行分析
if (require.main === module) {
  // 解析命令行参数
  const args = process.argv.slice(2);
  let workflowName = null;
  let runId = null;
  let jobId = null;
  
  let createIssue = false;
  let reportPath = null;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--workflow' && i + 1 < args.length) {
      workflowName = args[i + 1];
      i++;
    } else if (args[i] === '--run' && i + 1 < args.length) {
      runId = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--job' && i + 1 < args.length) {
      jobId = parseInt(args[i + 1]);
      i++;
    } else if (args[i] === '--create-issue' && i + 1 < args.length) {
      createIssue = true;
      reportPath = args[i + 1];
      i++;
    } else if (!workflowName && !runId && !jobId && !createIssue) {
      // 如果没有指定参数名，尝试解析为工作流名称
      workflowName = args[i];
    }
  }
  
  // 重定向console.log到文件
  const fs = require('fs');
  const logDir = './logs';
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  
  const logFileName = `analysis_${new Date().toISOString().replace(/[:.]/g, '-')}.log`;
  const logFilePath = `${logDir}/${logFileName}`;
  const logStream = fs.createWriteStream(logFilePath, { flags: 'a' });
  
  // 保存原始console.log
  const originalLog = console.log;
  
  // 重定义console.log
  console.log = function(...args) {
    // 调用原始console.log
    originalLog.apply(console, args);
    
    // 将内容写入文件
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg, null, 2) : arg).join(' ');
    logStream.write(message + '\n');
  };
  
  console.log(`Analysis started. Log file: ${logFilePath}`);
  
  const analyzer = new WorkflowTestAnalyzer(workflowName, runId);
  
  if (createIssue) {
    analyzer.createIssueFromReport(reportPath)
      .catch(console.error)
      .finally(() => {
        // 先保存原始的console.log
        const tempLog = console.log;
        // 临时恢复console.log为原始函数，避免在关闭流后写入
        console.log = originalLog;
        console.log(`Issue creation completed. Log saved to: ${logFilePath}`);
        // 关闭日志流
        logStream.end();
      });
  } else if (jobId) {
    analyzer.analyzeJobDirectly(jobId)
      .catch(console.error)
      .finally(() => {
        // 先保存原始的console.log
        const tempLog = console.log;
        // 临时恢复console.log为原始函数，避免在关闭流后写入
        console.log = originalLog;
        console.log(`Analysis completed. Log saved to: ${logFilePath}`);
        // 关闭日志流
        logStream.end();
      });
  } else {
    analyzer.analyzeWorkflowTests()
      .catch(console.error)
      .finally(() => {
        // 先保存原始的console.log
        const tempLog = console.log;
        // 临时恢复console.log为原始函数，避免在关闭流后写入
        console.log = originalLog;
        console.log(`Analysis completed. Log saved to: ${logFilePath}`);
        // 关闭日志流
        logStream.end();
      });
  }
}

module.exports = WorkflowTestAnalyzer;