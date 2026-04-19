import {
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class SynaptApi implements ICredentialType {
	name = 'synaptApi';
	displayName = 'Synapt API';
	documentationUrl = 'https://synapt.dev/guide.html';
	properties: INodeProperties[] = [
		{
			displayName: 'Server URL',
			name: 'serverUrl',
			type: 'string',
			default: 'http://localhost:8417',
			placeholder: 'http://localhost:8417',
			description: 'URL of the synapt recall MCP server (HTTP transport)',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description: 'Optional API key for authenticated synapt instances',
		},
	];
}
